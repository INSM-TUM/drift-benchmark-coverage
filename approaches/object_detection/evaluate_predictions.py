import os
import argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import ast
import re

import utils.config as cfg
import utils.utilities as utils
import utils.evaluation as eval
from utils.evaluation import (
    get_log_matching, get_drift_info, get_traces_per_log,
    get_log_info, get_true_changepoints_and_classes, get_evaluation_metrics,
    save_results, print_measures, plot_classification_report
)

def evaluate_predictions(pred_csv_path, data_dir, output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(pred_csv_path)
    
    # Load predictions
    results_df = pd.read_csv(pred_csv_path, index_col=0)
    
    drift_info = get_drift_info(data_dir)
    traces_per_log = get_traces_per_log(data_dir)
    log_matching = None
    if os.path.isfile(os.path.join(data_dir, "log_matching.csv")):
        log_matching = get_log_matching(data_dir)

    eval_results = defaultdict(lambda: defaultdict(dict))
    
    for image_name, row in tqdm(results_df.iterrows(), desc="Evaluating predictions", total=len(results_df)):
        image_name_str = str(image_name)
        if log_matching is not None:
            # Handle float image names if they were parsed as such
            image_id = int(float(image_name_str.split(".")[0]))
            log_name = log_matching.loc[log_matching["image_id"] == image_id, "log_name"].iloc[0]
        else:
            log_name = image_name_str
            if not log_name.endswith(".xes"):
                log_name += ".xes"
            
        pred_change_points = ast.literal_eval(row["Detected Changepoints"])
        pred_change_points = [(int(float(cp[0])), int(float(cp[1]))) for cp in pred_change_points]
        y_pred_category = ast.literal_eval(row["Detected Drift Types"])
        
        log_info = get_log_info(log_name, drift_info)
        
        true_change_points, y_true_category = get_true_changepoints_and_classes(log_info, n_traces=traces_per_log.get(log_name))
        
        for lag_factor in cfg.RELATIVE_LAG:
            metrics, y_pred_sorted, y_true_sorted = get_evaluation_metrics(
                y_true=true_change_points,
                y_pred=pred_change_points,
                y_true_label=y_true_category,
                y_pred_label=y_pred_category,
                factor=lag_factor,
                number_of_traces=traces_per_log[log_name])

            str_lag = f"lag_{lag_factor}"
            eval_results[str_lag][log_name] = {
                 "Detected Changepoints": pred_change_points,
                 "Actual Changepoints": true_change_points,
                 "Predicted Drift Types": y_pred_category,
                 "Actual Drift Types": y_true_category,
                 "F1-Score": metrics["f1"],
                 "Precision": metrics["precision"],
                 "Recall": metrics["recall"],
                 "Average Lag": metrics["lag"],
                 "y_pred_sorted": y_pred_sorted,
                 "y_true_sorted": y_true_sorted,
                 "n_traces": traces_per_log[log_name]
            }
            
    eval_results = dict(eval_results)
    for lag_factor in cfg.RELATIVE_LAG:
        str_lag = f"lag_{lag_factor}"
        results_df = save_results(eval_results[str_lag], lag_factor, output_dir)
        print_measures(results_df, traces_per_log, lag_factor, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("evaluate_predictions")
    parser.add_argument("--pred-csv", dest="pred_csv", required=True, type=str,
                        help="Path to prediction_results.csv")
    parser.add_argument("--data-dir", dest="data_dir", required=True, type=str,
                        help="Directory containing the log files and drift_info.csv")
    parser.add_argument("--output-dir", dest="output_dir", required=False, type=str,
                        default=None, help="Directory to save the generated evaluation_results_general_{lag}_lag.csv")
    args = parser.parse_args()
    
    evaluate_predictions(args.pred_csv, args.data_dir, args.output_dir)
