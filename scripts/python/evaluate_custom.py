import os
import sys
import json
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from typing import Dict, List, Tuple
import argparse

# Parse arguments to get optional run directory
parser = argparse.ArgumentParser(description="Evaluate on custom datasets.")
parser.add_argument("--base_rep", dest="base_rep",
                    help="Base representation (e.g., 'arm' or 'dfg')",
                    default="arm",
                    type=str)
args = parser.parse_args()

# Resolve root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

import src.config as cfg
import src.utilities as utils
import src.preprocessing as pp
import src.evaluation as eval
import src.cdrift_evaluation as cdrift
import src.prediction as pred_lib

def safe_filter_complete_events(log):
    """Filters complete events, falling back to original log if the result is empty."""
    filtered = utils.filter_complete_events(log)
    if len(filtered) == 0:
        return log
    return filtered

def parse_custom_drift_info(csv_path: str, log_traces: Dict[str, int]) -> Dict[str, List[Tuple[int, int]]]:
    """
    Parses CDLG style drift_info.csv file from custom datasets.
    Extracts relative positions in parentheses from change_start and change_end
    and maps them to absolute trace indices.
    """
    df = pd.read_csv(csv_path, sep=';')
    
    # Group by log_name and drift_or_noise_id
    grouped = df.groupby(['log_name', 'drift_or_noise_id'])
    
    gt_map = {}
    
    for (log_name, drift_id), group_df in grouped:
        total_traces = log_traces.get(log_name)
        if total_traces is None or total_traces <= 0:
            continue
            
        start_rel = None
        end_rel = None
        start_trace = None
        end_trace = None
        
        # Find change_start and change_end rows
        start_row = group_df[group_df['drift_sub_attribute'] == 'change_start']
        end_row = group_df[group_df['drift_sub_attribute'] == 'change_end']
        
        if not start_row.empty:
            val_str = str(start_row.iloc[0]['value'])
            # First try exact trace index format e.g. [2500]
            exact_match = re.search(r'\[(\d+)\]', val_str)
            if exact_match:
                start_trace = int(exact_match.group(1))
            else:
                match = re.search(r'\(([^)]+)\)', val_str)
                if match:
                    start_rel = float(match.group(1))
                
        if not end_row.empty:
            val_str = str(end_row.iloc[0]['value'])
            if val_str != 'N/A' and val_str != 'nan':
                exact_match = re.search(r'\[(\d+)\]', val_str)
                if exact_match:
                    end_trace = int(exact_match.group(1))
                else:
                    match = re.search(r'\(([^)]+)\)', val_str)
                    if match:
                        end_rel = float(match.group(1))
                    
        if start_trace is None and start_rel is not None:
            start_trace = int(start_rel * total_traces)
        if end_trace is None and end_rel is not None:
            end_trace = int(end_rel * total_traces)
        if end_trace is None and start_trace is not None:
            end_trace = start_trace
            
        if start_trace is not None:
                
            if log_name not in gt_map:
                gt_map[log_name] = []
            gt_map[log_name].append((start_trace, end_trace))
            
    return gt_map

def preprocess_and_evaluate_dataset(dataset_name: str, dataset_dir: str, 
                                    model, model_fn, output_base_dir: str) -> List[Dict]:
    """Runs the complete preprocess-predict-evaluate pipeline for a single custom dataset."""
    print(f"\n================ Processing Custom Dataset: {dataset_name.upper()} ================")
    
    preprocessed_dir = os.path.join(output_base_dir, dataset_name, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    if getattr(cfg, 'AUTO_RUN_PREPROCESSING', False):
        print(f"Step 1: Preprocessing event logs using generic pipeline...")
        pp.process_generic_directory(
            input_dir=dataset_dir,
            output_dir=preprocessed_dir,
            create_tfrecords=False
        )
    else:
        print(f"Step 1: Skipping preprocessing (AUTO_RUN_PREPROCESSING=False). Using existing data in {preprocessed_dir}")
    
    # Locate the nested winsim experiment directory
    import glob
    winsim_dirs = glob.glob(os.path.join(preprocessed_dir, "winsim", "experiment_*"))
    valid_dirs = [d for d in winsim_dirs if os.path.isfile(os.path.join(d, "window_info.json"))]
    
    if not valid_dirs:
        print(f"Error: Could not find completed winsim experiment folder in {preprocessed_dir}")
        return []
    actual_preprocessed_dir = sorted(valid_dirs)[-1]
    
    # Load metadata
    window_info_path = os.path.join(actual_preprocessed_dir, "window_info.json")
    with open(window_info_path, "r", encoding='utf-8') as f:
        window_info = json.load(f)
        
    log_traces_path = os.path.join(actual_preprocessed_dir, "number_of_traces.json")
    with open(log_traces_path, "r", encoding='utf-8') as f:
        log_traces = json.load(f)
        
    # 2. Load ground truth
    print("Step 2: Parsing ground truth drift info...")
    drift_info_path = os.path.join(dataset_dir, "drift_info.csv")
    if not os.path.exists(drift_info_path):
        print(f"Warning: drift_info.csv not found at {drift_info_path}. Cannot evaluate.")
        return []
    gt_map = parse_custom_drift_info(drift_info_path, log_traces)
    
    # 3. Prediction & Evaluation
    print("Step 3: Running predictions and matching metrics...")
    category_index, _ = utils.get_ex_decoder()
    eval_results = []
    
    threshold = cfg.EVAL_THRESHOLD
    targetsize = 256
    n_windows = cfg.N_WINDOWS
    
    # Load log_matching to map image_id -> log_name
    log_matching_path = os.path.join(actual_preprocessed_dir, "log_matching.csv")
    img_to_log = {}
    if os.path.exists(log_matching_path):
        lm_df = pd.read_csv(log_matching_path)
        if "log_name" in lm_df.columns:
            for _, row in lm_df.iterrows():
                img_to_log[str(row["image_id"])] = row["log_name"]
        elif lm_df.columns[0] == "Unnamed: 0":
            for _, row in lm_df.iterrows():
                img_to_log[str(row["image_id"])] = row["Unnamed: 0"]
                
    ext = ".png" if getattr(cfg, 'COLOR', 'grayscale') == "grayscale" else ".jpg"
    images = [f for f in os.listdir(actual_preprocessed_dir) if f.endswith(ext)]
    
    for img_file in tqdm(images, desc=f"Predicting & Evaluating ({dataset_name})"):
        img_id = img_file.split(".")[0]
        
        # Re-construct original log_name with extension
        log_name = img_to_log.get(img_id, img_id)
        if not log_name.endswith('.xes'):
            log_name = f"{log_name}.xes"
            
        log_name_base = log_name.split('.')[0]
        
        if log_name not in gt_map:
            print(f"Warning: {log_name} not found in ground truth map for custom dataset!")
            continue
            
        known_cps = gt_map[log_name]
        
        n_traces = log_traces[log_name]
        
        # Load and prepare image
        img_path = os.path.join(actual_preprocessed_dir, img_file)
        image = utils.load_image(img_path)
        image = utils.build_inputs_for_object_detection(image, cfg.IMAGE_SIZE)
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(image, dtype=tf.uint8)
        
        # Model Prediction
        result = model_fn(image)
        scores = result['detection_scores'][0].numpy()
        bbox_pred = result['detection_boxes'][0].numpy()
        bbox_pred = bbox_pred[scores > threshold]
        
        y_pred = result['detection_classes'][0].numpy().astype(int)
        y_pred = y_pred[scores > threshold]
        
        y_pred_category = eval.get_predicted_classes(y_pred, category_index)
        
        # Save visualization with predicted bounding boxes
        viz_dir = os.path.join(output_base_dir, dataset_name, "predictions_viz")
        os.makedirs(viz_dir, exist_ok=True)
        pred_lib.visualize_prediction(
            path=viz_dir,
            image=image,
            image_name=log_name_base,
            bbox_pred=bbox_pred,
            y_pred=y_pred,
            score=scores[scores > threshold],
            encoding="winsim"
        )
        
        # Scale bounding boxes to N_WINDOWS
        bbox_pred = bbox_pred / targetsize * n_windows
        
        log_window_info = window_info[log_name]
        
        # Convert predicted bboxes to trace indices
        pred_change_points = eval.get_changepoints_trace_idx_winsim(
            bbox_pred, y_pred_category, log_window_info)
            
        # Compute metrics for different lag configurations
        # 1. Absolute lag of 200 traces
        tp_200, fp_200, fn_200, prec_200, rec_200, f1_200, avg_lag_200 = compute_metrics(
            detected_drifts=pred_change_points, actual_drifts=known_cps, lag=200)
        
        # 2. Relative lags: 1%, 2.5% and 5%
        rel_metrics = {}
        for factor in [0.01, 0.025, 0.05]:
            lag_val = int(factor * n_traces)
            tp_rel, fp_rel, fn_rel, prec_rel, rec_rel, f1_rel, avg_lag_rel = compute_metrics(
                detected_drifts=pred_change_points, actual_drifts=known_cps, lag=lag_val)
            
            rel_metrics.update({
                f"tp_{factor}": tp_rel,
                f"fp_{factor}": fp_rel,
                f"fn_{factor}": fn_rel,
                f"precision_{factor}": prec_rel,
                f"recall_{factor}": rec_rel,
                f"f1_{factor}": f1_rel,
                f"avg_lag_{factor}": avg_lag_rel
            })
            
        result_row = {
            "dataset": dataset_name,
            "log_name": log_name,
            "n_traces": n_traces,
            "true_changepoints": [cp[0] for cp in known_cps],
            "detected_changepoints": pred_change_points,
            "tp_200": tp_200,
            "fp_200": fp_200,
            "fn_200": fn_200,
            "precision_200": prec_200,
            "recall_200": rec_200,
            "f1_200": f1_200,
            "avg_lag_200": avg_lag_200,
            **rel_metrics
        }
        eval_results.append(result_row)
        
    if eval_results:
        results_df = pd.DataFrame(eval_results)
        results_df.to_csv(os.path.join(output_base_dir, dataset_name, "evaluation_details.csv"), index=False)
        
    return eval_results

def main():
    
    # Update config overrides based on parsed arguments
    cfg.recalculate_derived_configs(args.base_rep)
    
    model_path = cfg.TRAINED_MODEL_PATH
    output_base_dir = cfg.CUSTOM_EVAL_OUTPUT_DIR
    custom_eval_root = cfg.CUSTOM_EVAL_ROOT
    
    # Check model path exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}!")
        sys.exit(1)
        
    
    # Load model
    print(f"Loading pre-trained model from {model_path}...")
    model = tf.saved_model.load(model_path)
    model_fn = model.signatures['serving_default']
    
    # Iterate through all custom subdirectories
    subdirs = [d for d in os.listdir(custom_eval_root) 
               if os.path.isdir(os.path.join(custom_eval_root, d))]
               
    print(f"Found custom datasets: {subdirs}")
    
    all_dataset_results = []
    
    for subdir in subdirs:
        dataset_dir = os.path.join(custom_eval_root, subdir)
        results = preprocess_and_evaluate_dataset(subdir, dataset_dir, model, model_fn, output_base_dir)
        all_dataset_results.extend(results)
        
    if not all_dataset_results:
        print("No evaluation results computed.")
        return
        
    # Compile aggregated summary
    summary_df = pd.DataFrame(all_dataset_results)
    
    summary_rows = []
    datasets = list(summary_df["dataset"].unique()) + ["Overall"]
    for ds in datasets:
        if ds == "Overall":
            ds_df = summary_df
        else:
            ds_df = summary_df[summary_df["dataset"] == ds]
            
        if ds_df.empty:
            continue
            
        # Micro-averaged metrics for 200 lag
        total_tp = ds_df["tp_200"].sum()
        total_fp = ds_df["fp_200"].sum()
        total_fn = ds_df["fn_200"].sum()
        
        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        avg_lag = ds_df["avg_lag_200"].mean(skipna=True)
        
        summary_rows.append({
            "Dataset": ds,
            "Logs Count": len(ds_df),
            "Total TP": total_tp,
            "Total FP": total_fp,
            "Total FN": total_fn,
            "Precision (200 lag)": round(prec, 4),
            "Recall (200 lag)": round(rec, 4),
            "F1-Score (200 lag)": round(f1, 4),
            "Mean Lag (200 lag)": round(avg_lag, 2) if not pd.isna(avg_lag) else "N/A"
        })
        
    summary_results_df = pd.DataFrame(summary_rows)
    summary_results_df.to_csv(os.path.join(output_base_dir, "custom_evaluation_summary.csv"), index=False)
    
    print("\n================ CUSTOM EVALUATION SUMMARY (200 LAG) ================")
    print(summary_results_df.to_string(index=False))

    # Calculate global metrics
    global_rows = []
    lags = [200, 0.01, 0.025, 0.05]
    for lag in lags:
        if lag == 200:
            tp_col, fp_col, fn_col = "tp_200", "fp_200", "fn_200"
        else:
            tp_col, fp_col, fn_col = f"tp_{lag}", f"fp_{lag}", f"fn_{lag}"
            
        total_tp = summary_df[tp_col].sum()
        total_fp = summary_df[fp_col].sum()
        total_fn = summary_df[fn_col].sum()
        
        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        global_rows.append({
            "lag": lag,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "n_logs": len(summary_df)
        })
        
    global_df = pd.DataFrame(global_rows)
    global_df.to_csv(os.path.join(output_base_dir, "global_metrics.csv"), index=False)
    print("\n================ GLOBAL METRICS ================")
    print(global_df.to_string(index=False))

    print(f"\nAll custom evaluations completed! Summaries saved to {output_base_dir}/")

if __name__ == "__main__":
    main()
