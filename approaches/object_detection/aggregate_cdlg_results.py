import os
import ast
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

try:
    from utils.cdrift_original import getTP_FP
except ImportError:
    try:
        from approaches.object_detection.utils.cdrift_original import getTP_FP
    except ImportError:
        def getTP_FP(detected: List[int], known: List[int], lag: int) -> Tuple[int, int]:
            if not detected or not known:
                return 0, len(detected)
            candidate_pairs = []
            for d_idx, d in enumerate(detected):
                for k_idx, k in enumerate(known):
                    dist = abs(d - k)
                    if dist <= lag:
                        candidate_pairs.append((dist, d_idx, k_idx))
            candidate_pairs.sort(key=lambda x: x[0])
            used_d, used_k, tp_count = set(), set(), 0
            for dist, d_idx, k_idx in candidate_pairs:
                if d_idx not in used_d and k_idx not in used_k:
                    used_d.add(d_idx)
                    used_k.add(k_idx)
                    tp_count += 1
            return tp_count, len(detected) - tp_count


def is_val_empty(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, (list, tuple, np.ndarray)):
        return len(val) == 0
    if pd.isna(val) is True:
        return True
    s_val = str(val).strip()
    return s_val == "" or s_val == "[]"


def parse_cp_list(val: Any) -> List[int]:
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


def parse_list(val: Any) -> List[Any]:
    if is_val_empty(val):
        return []
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return []
    elif isinstance(val, (list, tuple, np.ndarray)):
        return list(val)
    return []


def run_cdlg_aggregation():
    base_dir = "results"
    methods = ["arm", "dfg", "dfg_io"]
    latencies = [0.01, 0.025, 0.05]

    noise_info_path = os.path.join(base_dir, "input_cdlg", "log_noise_info.csv")
    noise_dict = {}
    if os.path.exists(noise_info_path):
        noise_df = pd.read_csv(noise_info_path)
        # Drop duplicates by log_name taking max value
        noise_df_clean = noise_df.groupby("log_name")["value"].max().to_dict()
        noise_dict = noise_df_clean

    overall_rows = []
    noise_rows = []
    drift_type_rows = []

    for method in methods:
        csv_path = os.path.join(base_dir, f"cdlg_predictions_{method}", "combined_results.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found. Skipping method {method}.")
            continue

        df = pd.read_csv(csv_path)

        # 1. Overall accuracy across latencies
        for lat in latencies:
            tp_sum, fp_sum, fn_tp_sum = 0, 0, 0
            for _, row in df.iterrows():
                log_size = row.get("log_size", 0)
                log_size = int(log_size) if pd.notna(log_size) and log_size else 0
                lag_acc = int(round(log_size * lat))

                actual_cps = parse_cp_list(row["actual_cp"])
                predicted_cps = parse_cp_list(row["predicted_cp"])

                tp, fp = getTP_FP(predicted_cps, actual_cps, lag_acc)
                fn_tp = len(actual_cps)

                tp_sum += tp
                fp_sum += fp
                fn_tp_sum += fn_tp

            precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
            recall = tp_sum / fn_tp_sum if fn_tp_sum > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            overall_rows.append({
                "method": method,
                "lag": lat,
                "tp": tp_sum,
                "fp": fp_sum,
                "fn": fn_tp_sum - tp_sum,
                "act_len": fn_tp_sum,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })

        # 2. Noise impact at 5% latency (lat = 0.05)
        lat = 0.05
        noise_groups = {0.0: {"tp": 0, "fp": 0, "fn_tp": 0},
                        0.3: {"tp": 0, "fp": 0, "fn_tp": 0},
                        0.6: {"tp": 0, "fp": 0, "fn_tp": 0}}

        for _, row in df.iterrows():
            log_name = str(row["log_name"])
            noise_val = noise_dict.get(log_name, 0.0)
            # Match nearest noise bucket (0.0, 0.3, 0.6)
            if noise_val >= 0.5:
                bucket = 0.6
            elif noise_val >= 0.2:
                bucket = 0.3
            else:
                bucket = 0.0

            log_size = row.get("log_size", 0)
            log_size = int(log_size) if pd.notna(log_size) and log_size else 0
            lag_acc = int(round(log_size * lat))

            actual_cps = parse_cp_list(row["actual_cp"])
            predicted_cps = parse_cp_list(row["predicted_cp"])

            tp, fp = getTP_FP(predicted_cps, actual_cps, lag_acc)
            fn_tp = len(actual_cps)

            noise_groups[bucket]["tp"] += tp
            noise_groups[bucket]["fp"] += fp
            noise_groups[bucket]["fn_tp"] += fn_tp

        for n_val, counts in noise_groups.items():
            tp = counts["tp"]
            fp = counts["fp"]
            fn_tp = counts["fn_tp"]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / fn_tp if fn_tp > 0 else 0.0
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

            noise_rows.append({
                "method": method,
                "lag": lat,
                "noise": n_val,
                "tp": tp,
                "fp": fp,
                "fn": fn_tp - tp,
                "act_len": fn_tp,
                "precision": p,
                "recall": r,
                "f1": f1
            })

        # 3. Drift Type Breakdown at 5% latency (lat = 0.05)
        drift_types = ["no_drifts", "sudden", "gradual", "incremental", "recurring"]
        dt_counts = {dt: {"tp": 0, "fp": 0, "fn_tp": 0} for dt in drift_types}

        for _, row in df.iterrows():
            log_size = row.get("log_size", 0)
            log_size = int(log_size) if pd.notna(log_size) and log_size else 0
            lag_acc = int(round(log_size * lat))

            actual_cps = parse_cp_list(row["actual_cp"])
            predicted_cps = parse_cp_list(row["predicted_cp"])
            act_types = parse_list(row["actual_drift_type"])
            pred_types = parse_list(row["predicted_drift_type"])

            if len(actual_cps) == 0:
                dt_counts["no_drifts"]["fn_tp"] += 1
                if len(predicted_cps) == 0:
                    dt_counts["no_drifts"]["tp"] += 1
                else:
                    dt_counts["no_drifts"]["fp"] += 1
            else:
                # Classify log primary drift type
                primary_type = "sudden"
                if any(t in act_types for t in ["recurring"]):
                    primary_type = "recurring"
                elif any(t in act_types for t in ["incremental"]):
                    primary_type = "incremental"
                elif any(t in act_types for t in ["gradual"]):
                    primary_type = "gradual"
                elif any(t in act_types for t in ["sudden"]):
                    primary_type = "sudden"

                tp, fp = getTP_FP(predicted_cps, actual_cps, lag_acc)
                fn_tp = len(actual_cps)

                dt_counts[primary_type]["tp"] += tp
                dt_counts[primary_type]["fp"] += fp
                dt_counts[primary_type]["fn_tp"] += fn_tp

        for dt in drift_types:
            tp = dt_counts[dt]["tp"]
            fp = dt_counts[dt]["fp"]
            fn_tp = dt_counts[dt]["fn_tp"]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / fn_tp if fn_tp > 0 else 0.0
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

            drift_type_rows.append({
                "method": method,
                "lag": lat,
                "drift_type": dt,
                "tp": tp,
                "fp": fp,
                "fn": fn_tp - tp,
                "act_len": fn_tp,
                "precision": p,
                "recall": r,
                "f1": f1
            })

    output_aggregated_dir = os.path.join(base_dir, "aggregated")
    os.makedirs(output_aggregated_dir, exist_ok=True)

    pd.DataFrame(overall_rows).to_csv(os.path.join(output_aggregated_dir, "cdlg_overall.csv"), index=False)
    pd.DataFrame(noise_rows).to_csv(os.path.join(output_aggregated_dir, "cdlg_noise.csv"), index=False)
    pd.DataFrame(drift_type_rows).to_csv(os.path.join(output_aggregated_dir, "cdlg_drift_types.csv"), index=False)

    print("CDLG Aggregation complete. CSVs saved to results/aggregated/")


if __name__ == "__main__":
    run_cdlg_aggregation()
