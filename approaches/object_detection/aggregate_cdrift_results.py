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


def run_cdrift_aggregation():
    base_dir = "results"
    methods = ["arm", "dfg", "dfg_io"]
    latencies = [0.01, 0.025, 0.05]
    windows_list = list(range(60, 210, 10))

    overall_rows = []
    windows_rows = []

    for method in methods:
        csv_path = os.path.join(base_dir, f"cdrift_predictions_{method}", "prediction_results_evaluated_all_windows.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found. Skipping method {method}.")
            continue

        df = pd.read_csv(csv_path)

        # 1. Overall metrics for N=70 across latencies
        w_df_70 = df[df["windows"] == 70] if "windows" in df.columns else df
        for lat in latencies:
            tp_sum, fp_sum, fn_tp_sum = 0, 0, 0
            for _, row in w_df_70.iterrows():
                log_size = row.get("log_size", 0)
                log_size = int(log_size) if pd.notna(log_size) and log_size else 0
                lag_acc = int(round(log_size * lat))

                actual_cps = parse_cp_list(row["actual_cp"])
                predicted_cps = parse_cp_list(row["detected_cp"])

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

        # 2. Window sensitivity metrics for lat = 0.05 across N in [60, 200]
        lat = 0.05
        for w in windows_list:
            w_sub = df[df["windows"] == w] if "windows" in df.columns else df
            tp_sum, fp_sum, fn_tp_sum = 0, 0, 0
            for _, row in w_sub.iterrows():
                log_size = row.get("log_size", 0)
                log_size = int(log_size) if pd.notna(log_size) and log_size else 0
                lag_acc = int(round(log_size * lat))

                actual_cps = parse_cp_list(row["actual_cp"])
                predicted_cps = parse_cp_list(row["detected_cp"])

                tp, fp = getTP_FP(predicted_cps, actual_cps, lag_acc)
                fn_tp = len(actual_cps)

                tp_sum += tp
                fp_sum += fp
                fn_tp_sum += fn_tp

            precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
            recall = tp_sum / fn_tp_sum if fn_tp_sum > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            windows_rows.append({
                "method": method,
                "windows": w,
                "lag": lat,
                "tp": tp_sum,
                "fp": fp_sum,
                "fn": fn_tp_sum - tp_sum,
                "act_len": fn_tp_sum,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })

    output_aggregated_dir = os.path.join(base_dir, "aggregated")
    os.makedirs(output_aggregated_dir, exist_ok=True)

    pd.DataFrame(overall_rows).to_csv(os.path.join(output_aggregated_dir, "cdrift_overall.csv"), index=False)
    pd.DataFrame(windows_rows).to_csv(os.path.join(output_aggregated_dir, "cdrift_windows.csv"), index=False)

    print("CDRIFT Aggregation complete. CSVs saved to results/aggregated/")


if __name__ == "__main__":
    run_cdrift_aggregation()
