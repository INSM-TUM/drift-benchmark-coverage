import os
import ast
import re
import json
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional

try:
    from utils.cdrift_original import getTP_FP
except ImportError:
    try:
        from approaches.object_detection.utils.cdrift_original import getTP_FP
    except ImportError:
        def getTP_FP(detected: List[int], known: List[int], lag: int) -> Tuple[int, int]:
            """
            Fallback implementation for getTP_FP if pulp or external module is unavailable.
            Uses greedy matching based on absolute distance <= lag.
            """
            if not detected or not known:
                return 0, len(detected)
            
            candidate_pairs = []
            for d_idx, d in enumerate(detected):
                for k_idx, k in enumerate(known):
                    dist = abs(d - k)
                    if dist <= lag:
                        candidate_pairs.append((dist, d_idx, k_idx))
            
            candidate_pairs.sort(key=lambda x: x[0])
            
            used_d = set()
            used_k = set()
            tp_count = 0
            
            for dist, d_idx, k_idx in candidate_pairs:
                if d_idx not in used_d and k_idx not in used_k:
                    used_d.add(d_idx)
                    used_k.add(k_idx)
                    tp_count += 1
                    
            fp_count = len(detected) - tp_count
            return tp_count, fp_count


def is_val_empty(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, (list, tuple, np.ndarray)):
        return len(val) == 0
    if pd.isna(val) is True:
        return True
    s_val = str(val).strip()
    return s_val == "" or s_val == "[]"


def parse_confidence(val: Any) -> List[float]:
    if is_val_empty(val):
        return []
    if isinstance(val, (list, tuple, np.ndarray)):
        return [float(x) for x in val]
    clean_val = str(val).replace("[", "").replace("]", "").strip()
    if not clean_val:
        return []
    return [float(x) for x in re.split(r"\s+", clean_val)]


def parse_safely(val: Any) -> Any:
    if is_val_empty(val):
        return []
    if isinstance(val, (list, tuple, np.ndarray)):
        return val
    try:
        return ast.literal_eval(val)
    except Exception:
        return []


def parse_cp_list(val: Any) -> List[int]:
    """
    Safely parses a string/list/tuple representation of changepoint tuples/lists
    and returns a sorted list of unique integer changepoints.
    """
    if is_val_empty(val):
        return []
    
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
        except Exception:
            return []
    else:
        parsed = val
        
    cps = []
    if isinstance(parsed, (list, tuple, np.ndarray)):
        for item in parsed:
            if isinstance(item, (list, tuple, np.ndarray)):
                for x in item:
                    try:
                        cps.append(int(x))
                    except (ValueError, TypeError):
                        pass
            else:
                try:
                    cps.append(int(item))
                except (ValueError, TypeError):
                    pass
    return sorted(list(set(cps)))


def extract_drift_info(di_df: pd.DataFrame) -> Dict[str, Dict[str, List]]:
    di_dict: Dict[str, Dict[str, List]] = {}
    grouped = di_df.groupby(["log_name", "drift_or_noise_id"])

    for (log_name, drift_id), group_df in grouped:
        if log_name not in di_dict:
            di_log = {'drift_type': [], 'actual_cp': []}
            di_dict[log_name] = di_log
        else:
            di_log = di_dict[log_name]

        drift_list = group_df["drift_type"].tolist()
        if any(d in drift_list for d in ["sudden", "gradual"]):
            if len(drift_list) > 1:
                raise ValueError("There are multiple drifts in single drift")
            drift = group_df.head(1)
            di_log['drift_type'].append(drift['change_type'].tolist()[0])
            di_log['actual_cp'].append(tuple(ast.literal_eval(drift['drift_traces_index'].tolist()[0])))
        elif any(d in drift_list for d in ["recurring", "incremental"]):
            drift_type = group_df['drift_type'].tolist()[0]
            min_cp = 200000
            max_cp = 0
            for row in group_df.itertuples():
                di_log['drift_type'].append(row.change_type)
                cp = tuple(ast.literal_eval(row.drift_traces_index))
                di_log['actual_cp'].append(cp)
                min_cp = min(min_cp, cp[0])
                max_cp = max(max_cp, cp[-1])
            di_log['drift_type'].append(drift_type)
            di_log['actual_cp'].append((min_cp, max_cp))
        else:
            raise ValueError(f"Unknown drift type: {group_df['drift_type'].tolist()}")
    return di_dict


def extract_predictions(pred_df: pd.DataFrame, confidence_threshold: float = 0.5) -> Dict[str, Dict[str, List]]:
    di_dict: Dict[str, Dict[str, List]] = {}
    for row in pred_df.itertuples():
        log_name = row.log_name + ".xes" if not str(row.log_name).endswith(".xes") else row.log_name

        if log_name not in di_dict:
            di_log = {'detected_drift_type': [], 'detected_cp': []}
            di_dict[log_name] = di_log
        else:
            di_log = di_dict[log_name]

        for cp, drift_type, confidence in zip(
            row.Detected_Changepoints,
            row.Detected_Drift_Types,
            row.Prediction_Confidence,
        ):
            if confidence < confidence_threshold:
                continue

            start_cp = int(cp[0])
            end_cp = int(cp[1])
            di_log['detected_drift_type'].append(drift_type)
            di_log['detected_cp'].append((start_cp, end_cp))

    return di_dict


def combine_results(actual_drifts: Dict, predicted_drifts: Dict, traces_dict: Optional[Dict] = None) -> pd.DataFrame:
    rows = []
    all_logs = sorted(set(actual_drifts.keys()) | set(predicted_drifts.keys()))

    for log_name in all_logs:
        act = actual_drifts.get(log_name, {})
        pred = predicted_drifts.get(log_name, {})
        log_size = traces_dict.get(log_name, None) if traces_dict else None

        rows.append({
            "log_name": log_name,
            "log_size": log_size,
            "actual_drift_type": act.get("drift_type", []),
            "actual_cp": act.get("actual_cp", []),
            "predicted_drift_type": pred.get("detected_drift_type", []),
            "predicted_cp": pred.get("detected_cp", []),
        })

    return pd.DataFrame(rows)


def calculate_line_by_line_metrics(
    combined_df: pd.DataFrame,
    lag: float = 0.05,
    traces_dict: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Calculates TP, FP, and FN_TP scores for each line given a relative lag (percentage of log_size).
    """
    df = combined_df.copy()
    lag_pct = lag / 100.0 if lag > 1.0 else lag

    lag_indices_list = []
    tp_list = []
    fp_list = []
    fn_tp_list = []

    for _, row in df.iterrows():
        log_name = str(row["log_name"])
        log_size = row.get("log_size", None)
        if (pd.isna(log_size) is True or log_size is None) and traces_dict:
            log_size = traces_dict.get(log_name, 0)
        
        log_size = int(log_size) if log_size else 0
        lag_acc = int(round(log_size * lag_pct))

        actual_cps = parse_cp_list(row["actual_cp"])
        predicted_cps = parse_cp_list(row["predicted_cp"])

        tp, fp = getTP_FP(predicted_cps, actual_cps, lag_acc)
        fn_tp = len(actual_cps)

        lag_indices_list.append(lag_acc)
        tp_list.append(tp)
        fp_list.append(fp)
        fn_tp_list.append(fn_tp)

    df["lag"] = lag_pct
    df["lag_indices"] = lag_indices_list
    df["TP"] = tp_list
    df["FP"] = fp_list
    df["FN_TP"] = fn_tp_list

    return df


def aggregate_total_metrics(evaluated_df: pd.DataFrame) -> Dict[str, float]:
    """
    Aggregates TP, FP, and FN_TP scores across the entire dataset to calculate overall Precision, Recall, and F1.
    """
    tp_sum = evaluated_df["TP"].sum()
    fp_sum = evaluated_df["FP"].sum()
    fn_tp_sum = evaluated_df["FN_TP"].sum()

    precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    recall = tp_sum / fn_tp_sum if fn_tp_sum > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp_sum": tp_sum,
        "fp_sum": fp_sum,
        "fn_tp_sum": fn_tp_sum,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate(
    prediction_results_path: str,
    drift_info_path: str = "data/test/drift_info.csv",
    number_of_traces_path: str = "data/test/number_of_traces.json",
    output_dir: Optional[str] = None,
    lag: float = 0.05,
    confidence_threshold: float = 0.5,
):
    prediction_results_path = os.path.abspath(os.path.expanduser(prediction_results_path))
    drift_info_path = os.path.abspath(os.path.expanduser(drift_info_path))
    number_of_traces_path = os.path.abspath(os.path.expanduser(number_of_traces_path))

    if output_dir is None:
        output_dir = os.path.dirname(prediction_results_path)
    else:
        output_dir = os.path.abspath(os.path.expanduser(output_dir))

def extract_drift_info_from_csv(drift_info_path: str) -> Dict[str, Dict[str, List]]:
    di_df = pd.read_csv(drift_info_path, sep=";", skipinitialspace=True)
    di_df.columns = di_df.columns.str.strip()
    di_df = di_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    if "drift_type" in di_df.columns and ("drift_traces_index" in di_df.columns or "change_trace_index" in di_df.columns):
        return extract_drift_info(di_df)

    di_dict: Dict[str, Dict[str, List]] = {}

    grouped = di_df.groupby("log_name")
    for log_name, group in grouped:
        log_key = log_name if str(log_name).endswith(".xes") else str(log_name) + ".xes"
        if log_key not in di_dict:
            di_dict[log_key] = {"drift_type": [], "actual_cp": []}

        drift_groups = group.groupby("drift_or_noise_id")
        for drift_id, d_group in drift_groups:
            dt_rows = d_group[d_group["drift_attribute"] == "drift_type"]
            if dt_rows.empty:
                dt_rows = d_group[d_group["drift_sub_attribute"] == "drift_type"]
            d_type = "sudden"
            if not dt_rows.empty:
                d_type = str(dt_rows["value"].iloc[0])

            cps = []
            for _, row in d_group.iterrows():
                sub_attr = str(row.get("drift_sub_attribute", ""))
                attr = str(row.get("drift_attribute", ""))
                val = str(row.get("value", ""))

                if "change_start" in sub_attr or "change_trace_index" in sub_attr or "change_info" in attr or "change_start" in val:
                    match = re.search(r"\[(\d+(?:\.\d+)?)\]", val)
                    if match:
                        cps.append(int(float(match.group(1))))
                    else:
                        digits = re.findall(r"\b\d+(?:\.\d+)?\b", val)
                        if digits:
                            cps.append(int(float(digits[-1])))

            if cps:
                cps = sorted(list(set(cps)))
                if len(cps) == 1:
                    cp_tuple = (cps[0], cps[0])
                else:
                    cp_tuple = (cps[0], cps[-1])
                
                di_dict[log_key]["drift_type"].append(d_type)
                di_dict[log_key]["actual_cp"].append(cp_tuple)

    return di_dict


def evaluate(
    prediction_results_path: str,
    drift_info_path: str,
    number_of_traces_path: str,
    output_dir: Optional[str] = None,
    lag: float = 0.05,
    confidence_threshold: float = 0.5,
):
    if output_dir is None:
        output_dir = os.path.dirname(prediction_results_path)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Read actual drift info
    actual_drifts = extract_drift_info_from_csv(drift_info_path)

    # 2. Read prediction results
    pred_df = pd.read_csv(prediction_results_path)
    pred_df.rename(columns={"Unnamed: 0": "log_name"}, inplace=True)
    pred_df.rename(columns={"Detected Changepoints": "Detected_Changepoints"}, inplace=True)
    pred_df.rename(columns={"Detected Drift Types": "Detected_Drift_Types"}, inplace=True)
    pred_df.rename(columns={"Prediction Confidence": "Prediction_Confidence"}, inplace=True)

    pred_df["Detected_Changepoints"] = pred_df["Detected_Changepoints"].apply(parse_safely)
    pred_df["Detected_Drift_Types"] = pred_df["Detected_Drift_Types"].apply(parse_safely)
    pred_df["Prediction_Confidence"] = pred_df["Prediction_Confidence"].apply(parse_confidence)

    predictions = extract_predictions(pred_df, confidence_threshold=confidence_threshold)

    # 3. Read number of traces
    traces_dict = {}
    if os.path.exists(number_of_traces_path):
        with open(number_of_traces_path, "r") as f:
            traces_dict = json.load(f)

    # 4. Combine results
    combined_df = combine_results(actual_drifts, predictions, traces_dict)
    combined_csv_path = os.path.join(output_dir, "combined_results.csv")
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Saved combined results to {combined_csv_path}")

    # 5. Evaluate line by line metrics with relative lag
    evaluated_df = calculate_line_by_line_metrics(combined_df, lag=lag, traces_dict=traces_dict)
    evaluated_csv_path = os.path.join(output_dir, "combined_results_evaluated.csv")
    evaluated_df.to_csv(evaluated_csv_path, index=False)
    print(f"Saved line-by-line evaluated metrics to {evaluated_csv_path}")

    # 6. Aggregate metrics
    metrics = aggregate_total_metrics(evaluated_df)
    print(f"Precision: {metrics['precision']}, Recall: {metrics['recall']}, F1: {metrics['f1']}")

    total_metrics_path = os.path.join(output_dir, "total_metrics.txt")
    with open(total_metrics_path, "w") as f:
        f.write(f"TP Sum: {metrics['tp_sum']}\n")
        f.write(f"FP Sum: {metrics['fp_sum']}\n")
        f.write(f"FN_TP Sum: {metrics['fn_tp_sum']}\n")
        f.write(f"Precision: {metrics['precision']}\n")
        f.write(f"Recall: {metrics['recall']}\n")
        f.write(f"F1: {metrics['f1']}\n")
        f.write(f"Lag: {lag}\n")

    print(f"Saved total metrics summary to {total_metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate evaluation results for CDLG drift detection predictions.")
    parser.add_argument(
        "-p", "--prediction_results",
        type=str,
        required=True,
        help="Path (absolute or relative) to prediction_results.csv file."
    )
    parser.add_argument(
        "-d", "--drift_info",
        type=str,
        default="data/test/drift_info.csv",
        help="Path to ground truth drift_info.csv file."
    )
    parser.add_argument(
        "-t", "--number_of_traces",
        type=str,
        default="data/test/number_of_traces.json",
        help="Path to number_of_traces.json file."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default=None,
        help="Output directory to save results. Defaults to the folder containing prediction_results.csv."
    )
    parser.add_argument(
        "-l", "--lag",
        type=float,
        default=0.05,
        help="Relative lag as a percentage of total log traces (e.g., 0.05 for 5%%). Default: 0.05."
    )
    parser.add_argument(
        "-c", "--confidence_threshold",
        type=float,
        default=0.5,
        help="Prediction confidence threshold. Default: 0.5."
    )

    args = parser.parse_args()

    evaluate(
        prediction_results_path=args.prediction_results,
        drift_info_path=args.drift_info,
        number_of_traces_path=args.number_of_traces,
        output_dir=args.output_dir,
        lag=args.lag,
        confidence_threshold=args.confidence_threshold,
    )
