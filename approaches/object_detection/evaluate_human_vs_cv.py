import os
import sys
import json
import ast
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
obj_detection_dir = os.path.join(repo_root, "approaches", "object_detection")

if obj_detection_dir not in sys.path:
    sys.path.insert(0, obj_detection_dir)

import calculate_cdlg_evaluation_results as eval_cdlg


def parse_human_changepoints(val: Any) -> List[tuple]:
    """Safely parse human prediction changepoint strings/lists into list of integer tuples."""
    if pd.isna(val) or val is None:
        return []
    if isinstance(val, (list, tuple)):
        raw = val
    else:
        s_val = str(val).strip()
        if not s_val or s_val == "[]":
            return []
        try:
            raw = ast.literal_eval(s_val)
        except Exception:
            return []

    res = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    start_cp = int(float(str(item[0])))
                    end_cp = int(float(str(item[1])))
                    res.append((start_cp, end_cp))
                except Exception:
                    pass
            elif isinstance(item, (int, float, str)):
                try:
                    cp_val = int(float(str(item)))
                    res.append((cp_val, cp_val))
                except Exception:
                    pass
    return res


def evaluate_human_predictions(
    human_csv_path: str,
    drift_info_path: str,
    num_traces_path: str,
    output_dir: str,
    lag: float = 0.05
) -> Dict[str, float]:
    """Evaluates human predictions against ground truth drift_info.csv."""
    actual_drifts = eval_cdlg.extract_drift_info_from_csv(drift_info_path)

    human_df = pd.read_csv(human_csv_path)
    human_df.rename(columns={"Unnamed: 0": "log_name"}, inplace=True)
    human_df.rename(columns={"Detected Changepoints": "Detected_Changepoints"}, inplace=True)
    human_df.rename(columns={"Detected Drift Types": "Detected_Drift_Types"}, inplace=True)

    human_predictions = {}
    for row in human_df.itertuples():
        raw_name = str(row.log_name)
        log_name = raw_name + ".xes" if not raw_name.endswith(".xes") else raw_name
        cps = parse_human_changepoints(row.Detected_Changepoints)

        drift_types = []
        if hasattr(row, "Detected_Drift_Types") and not pd.isna(row.Detected_Drift_Types):
            try:
                dt_raw = ast.literal_eval(str(row.Detected_Drift_Types))
                if isinstance(dt_raw, (list, tuple)):
                    drift_types = [str(x) for x in dt_raw]
            except Exception:
                pass
        if not drift_types:
            drift_types = ["sudden"] * len(cps)

        human_predictions[log_name] = {
            "detected_drift_type": drift_types,
            "detected_cp": cps
        }

    traces_dict = {}
    if os.path.exists(num_traces_path):
        with open(num_traces_path, "r", encoding="utf-8") as f:
            traces_dict = json.load(f)

    combined_df = eval_cdlg.combine_results(actual_drifts, human_predictions, traces_dict)
    combined_human_csv = os.path.join(output_dir, "combined_results_human.csv")
    combined_df.to_csv(combined_human_csv, index=False)

    evaluated_df = eval_cdlg.calculate_line_by_line_metrics(combined_df, lag=lag, traces_dict=traces_dict)
    evaluated_human_csv = os.path.join(output_dir, "combined_results_human_evaluated.csv")
    evaluated_df.to_csv(evaluated_human_csv, index=False)

    metrics = eval_cdlg.aggregate_total_metrics(evaluated_df)
    
    metrics_txt_path = os.path.join(output_dir, "human_total_metrics.txt")
    with open(metrics_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Precision: {metrics['precision']}\n")
        f.write(f"Recall: {metrics['recall']}\n")
        f.write(f"F1: {metrics['f1']}\n")
        f.write(f"TP: {metrics['tp_sum']}, FP: {metrics['fp_sum']}, FN_TP: {metrics['fn_tp_sum']}\n")

    return metrics


def run_human_vs_cv_evaluation():
    outputs_dir = os.path.join(repo_root, "outputs")
    custom_eval_dir = os.path.join(repo_root, "data", "custom_eval")

    representations = ["dfg", "dfg_io", "arm"]
    subdirs = sorted([d for d in os.listdir(custom_eval_dir) 
                      if os.path.isdir(os.path.join(custom_eval_dir, d)) and not d.startswith(".")])

    comparison_results = []

    for rep in representations:
        for subdir in subdirs:
            out_subdir = os.path.join(outputs_dir, f"custom_eval_predictions_{rep}", subdir)
            human_csv = os.path.join(out_subdir, "prediction_human.csv")
            drift_info_path = os.path.join(custom_eval_dir, subdir, "drift_info.csv")
            num_traces_path = os.path.join(out_subdir, "number_of_traces.json")

            if not os.path.exists(human_csv) or not os.path.exists(drift_info_path):
                continue

            h_metrics = evaluate_human_predictions(
                human_csv_path=human_csv,
                drift_info_path=drift_info_path,
                num_traces_path=num_traces_path,
                output_dir=out_subdir,
                lag=0.05
            )

            cv_prec, cv_rec, cv_f1 = 0.0, 0.0, 0.0
            cv_metrics_path = os.path.join(out_subdir, "total_metrics.txt")
            if os.path.exists(cv_metrics_path):
                with open(cv_metrics_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("Precision:"):
                            cv_prec = float(line.split(":")[1].strip())
                        elif line.startswith("Recall:"):
                            cv_rec = float(line.split(":")[1].strip())
                        elif line.startswith("F1:"):
                            cv_f1 = float(line.split(":")[1].strip())

            h_prec = h_metrics["precision"]
            h_rec = h_metrics["recall"]
            h_f1 = h_metrics["f1"]

            if h_f1 > cv_f1:
                winner = "Human"
            elif cv_f1 > h_f1:
                winner = "CV Model"
            else:
                winner = "Tie"

            comparison_results.append({
                "Representation": rep.upper(),
                "Dataset": subdir,
                "Human_Precision": round(h_prec, 4),
                "Human_Recall": round(h_rec, 4),
                "Human_F1": round(h_f1, 4),
                "CV_Precision": round(cv_prec, 4),
                "CV_Recall": round(cv_rec, 4),
                "CV_F1": round(cv_f1, 4),
                "Winner": winner
            })

    comp_df = pd.DataFrame(comparison_results)
    
    print("\n\n==========================================================================")
    print("        HUMAN PREDICTIONS vs. CV MODEL PREDICTIONS COMPARISON            ")
    print("==========================================================================")
    print(comp_df.to_string(index=False))

    comp_csv_path = os.path.join(outputs_dir, "human_vs_cv_comparison.csv")
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"\nSaved comparison summary table to: {comp_csv_path}")


if __name__ == "__main__":
    run_human_vs_cv_evaluation()
