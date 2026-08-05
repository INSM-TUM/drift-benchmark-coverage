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


def run_noise_update_and_aggregation():
    base_dir = "results"
    methods = ["arm", "dfg", "dfg_io"]
    lat = 0.05

    noise_summary_path = os.path.join(base_dir, "input_cdlg", "noise_summary.csv")
    if not os.path.exists(noise_summary_path):
        print(f"Error: {noise_summary_path} not found.")
        return

    noise_df = pd.read_csv(noise_summary_path)
    noise_dict = dict(zip(noise_df["log_name"], noise_df["noise"]))

    noise_rows = []

    for method in methods:
        method_dir = os.path.join(base_dir, f"cdlg_predictions_{method}")
        comb_csv_path = os.path.join(method_dir, "combined_results.csv")
        eval_csv_path = os.path.join(method_dir, "combined_results_evaluated.csv")

        if not os.path.exists(comb_csv_path):
            print(f"Warning: {comb_csv_path} not found. Skipping {method}.")
            continue

        df_comb = pd.read_csv(comb_csv_path)

        # 1. Add noise column to combined_results.csv
        df_comb["noise"] = df_comb["log_name"].map(noise_dict).fillna(0.0)
        df_comb.to_csv(comb_csv_path, index=False)
        print(f"Updated noise column in {comb_csv_path}")

        if os.path.exists(eval_csv_path):
            df_eval = pd.read_csv(eval_csv_path)
            df_eval["noise"] = df_eval["log_name"].map(noise_dict).fillna(0.0)
            df_eval.to_csv(eval_csv_path, index=False)

        # 2. Perform noise level aggregation for 5% latency (lat = 0.05)
        noise_buckets = [0.0, 0.3, 0.6]
        for n_val in noise_buckets:
            sub_df = df_comb[df_comb["noise"] == n_val]
            tp_sum, fp_sum, fn_tp_sum = 0, 0, 0

            for _, row in sub_df.iterrows():
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

            p = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
            r = tp_sum / fn_tp_sum if fn_tp_sum > 0 else 0.0
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

            noise_rows.append({
                "method": method,
                "lag": lat,
                "noise": n_val,
                "tp": tp_sum,
                "fp": fp_sum,
                "fn": fn_tp_sum - tp_sum,
                "act_len": fn_tp_sum,
                "precision": p,
                "recall": r,
                "f1": f1
            })

    output_aggregated_dir = os.path.join(base_dir, "aggregated")
    os.makedirs(output_aggregated_dir, exist_ok=True)
    out_noise_csv = os.path.join(output_aggregated_dir, "cdlg_noise.csv")
    pd.DataFrame(noise_rows).to_csv(out_noise_csv, index=False)
    print(f"Noise aggregation complete. Results saved to {out_noise_csv}")


if __name__ == "__main__":
    run_noise_update_and_aggregation()
