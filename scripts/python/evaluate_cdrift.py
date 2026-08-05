import os
import sys
import json
import ast
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from typing import Dict, List, Tuple
import argparse

# Parse arguments to get optional run directory
parser = argparse.ArgumentParser(description="Evaluate on CDrift datasets.")
parser.add_argument("--base_rep", dest="base_rep",
                    help="Base representation (e.g., 'arm' or 'dfg')",
                    default="arm",
                    type=str)
parser.add_argument("--n_windows", dest="n_windows",
                    help="Override N_WINDOWS at runtime (avoids modifying config.py).",
                    default=None,
                    type=int)
args = parser.parse_args()

# Resolve root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

import src.config as cfg
import src.utilities as utils
import src.preprocessing as pp
import src.evaluation as eval
import src.cdrift_evaluation as cdrift

def load_ground_truth(csv_path: str) -> Dict[str, List[Tuple[int, int]]]:
    """Loads ground truth change points from CSV and converts them to (cp, cp) tuples."""
    df = pd.read_csv(csv_path)
    gt_map = {}
    for _, row in df.iterrows():
        log_name = row['log_name']
        cp_str = row['change_points']
        try:
            cps = ast.literal_eval(cp_str)
            if isinstance(cps, (int, float)):
                cps = [int(cps)]
            else:
                cps = [int(x) for x in cps]
        except Exception:
            cps = []
        # Convert to tuple list for compatibility with 2D matching in cdrift_evaluation.py
        gt_map[log_name] = [(cp, cp) for cp in cps]
    return gt_map

def safe_filter_complete_events(log):
    """Filters complete events, falling back to original log if the result is empty."""
    filtered = utils.filter_complete_events(log)
    if len(filtered) == 0:
        return log
    return filtered

def preprocess_cdrift_logs(log_dir: str, output_dir: str):
    """Preprocesses XES logs in log_dir to ARM similarity matrix images using generic pipeline."""
    print(f"Starting generic preprocessing for CDrift logs in {log_dir}...")
    pp.process_generic_directory(
        input_dir=log_dir,
        output_dir=output_dir,
        create_tfrecords=False
    )
        
    print("Preprocessing completed successfully.")

def run_prediction_and_evaluation(preprocessed_dir: str, model_path: str, 
                                  gt_map: Dict[str, List[Tuple[int, int]]], output_eval_dir: str):
    """Runs predictions on similarity matrix images and evaluates against CDrift ground truth."""
    os.makedirs(output_eval_dir, exist_ok=True)
    
    print(f"Loading pre-trained model from {model_path}...")
    model = tf.saved_model.load(model_path)
    model_fn = model.signatures['serving_default']
    
    # Locate the nested winsim experiment directory
    import glob
    winsim_dirs = glob.glob(os.path.join(preprocessed_dir, "winsim", "experiment_*"))
    valid_dirs = [d for d in winsim_dirs if os.path.isfile(os.path.join(d, "window_info.json"))]
    
    if not valid_dirs:
        print(f"Error: Could not find completed winsim experiment folder in {preprocessed_dir}")
        return
    actual_preprocessed_dir = sorted(valid_dirs)[-1]
    
    # Load window_info.json
    win_info_path = os.path.join(actual_preprocessed_dir, "window_info.json")
    with open(win_info_path, "r") as f:
        window_info = json.load(f)
        
    category_index, _ = utils.get_ex_decoder()
    
    # Load log_matching to map image_id -> log_name
    log_matching_path = os.path.join(actual_preprocessed_dir, "log_matching.csv")
    img_to_log = {}
    if os.path.exists(log_matching_path):
        lm_df = pd.read_csv(log_matching_path)
        # Ensure log_name is a column, or it's the index
        if "log_name" in lm_df.columns:
            for _, row in lm_df.iterrows():
                img_to_log[str(row["image_id"])] = row["log_name"]
        elif lm_df.columns[0] == "Unnamed: 0":
            for _, row in lm_df.iterrows():
                img_to_log[str(row["image_id"])] = row["Unnamed: 0"]
    
    images = {}
    ext = ".png" if getattr(cfg, 'COLOR', 'grayscale') == "grayscale" else ".jpg"
    for file in os.listdir(actual_preprocessed_dir):
        if file.endswith(ext):
            img_id = file.split(".")[0]
            # Map back to original log name (stripping .xes if present)
            original_log_name = img_to_log.get(img_id, img_id)
            if original_log_name.endswith('.xes'):
                original_log_name = original_log_name[:-4]
            images[original_log_name] = os.path.join(actual_preprocessed_dir, file)
            
    print(f"Found {len(images)} preprocessed images. Running prediction and evaluation...")
    
    eval_results = []
    threshold = cfg.EVAL_THRESHOLD
    targetsize = 256
    n_windows = cfg.N_WINDOWS
    
    for log_name, img_path in tqdm(images.items(), desc="Predicting & Evaluating"):
        if log_name not in gt_map:
            print(f"Warning: {log_name} not found in ground truth map!")
            continue
            
        known_cps = gt_map[log_name]
        
        # Load and prepare image
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
        
        # Scale bounding boxes to N_WINDOWS
        bbox_pred = bbox_pred / targetsize * n_windows
        
        log_window_info = window_info.get(log_name) or window_info.get(f"{log_name}.xes")
        if not log_window_info:
            print(f"Warning: No window info found for {log_name}")
            continue
        
        # Convert predicted bboxes to trace indices
        pred_change_points = eval.get_changepoints_trace_idx_winsim(
            bbox_pred, y_pred_category, log_window_info)
            
        # Convert all predictions to point-drifts (Sudden) by taking the center.
        # This prevents False Negatives when the model detects a drift but classifies it as Gradual.
        pred_change_points = [(round((cp[0] + cp[1]) / 2), round((cp[0] + cp[1]) / 2)) for cp in pred_change_points]
            
        # Get dataset type (Bose, Ostovar, Ceravolo) and trace count
        dataset_name = "Unknown"
        n_traces = 1000
        df_log = pd.read_csv(os.path.join(ROOT_DIR, "data/input_cdrift/log_size_info.csv"))
        match = df_log[df_log["log_name"] == log_name]
        if not match.empty:
            dataset_name = match.iloc[0]["dataset"]
            n_traces = int(match.iloc[0]["n_traces"])
            
        # 1. Absolute lag of 200 traces (standard CDrift metric)
        (tp_200, fp_200), assignments_200 = cdrift.getTP_FP(
            detected=pred_change_points, known=known_cps, lag=200)
        fn_200 = len(known_cps) - tp_200
        prec_200 = tp_200 / (tp_200 + fp_200) if (tp_200 + fp_200) > 0 else 0.0
        rec_200 = tp_200 / len(known_cps) if len(known_cps) > 0 else 0.0
        f1_200 = (2 * prec_200 * rec_200) / (prec_200 + rec_200) if (prec_200 + rec_200) > 0 else 0.0
        
        # Calculate lag
        avg_lag_200 = 0.0
        for (dc, ap) in assignments_200:
            dc_center = (dc[0] + dc[1]) / 2.0
            ap_center = (ap[0] + ap[1]) / 2.0
            avg_lag_200 += abs(dc_center - ap_center)
        avg_lag_200 = avg_lag_200 / len(assignments_200) if len(assignments_200) > 0 else np.nan
        
        # 2. Relative lags: 1%, 2.5% and 5%
        rel_metrics = {}
        for factor in [0.01, 0.025, 0.05]:
            lag_val = int(factor * n_traces)
            (tp_rel, fp_rel), assignments_rel = cdrift.getTP_FP(
                detected=pred_change_points, known=known_cps, lag=lag_val)
            fn_rel = len(known_cps) - tp_rel
            prec_rel = tp_rel / (tp_rel + fp_rel) if (tp_rel + fp_rel) > 0 else 0.0
            rec_rel = tp_rel / len(known_cps) if len(known_cps) > 0 else 0.0
            f1_rel = (2 * prec_rel * rec_rel) / (prec_rel + rec_rel) if (prec_rel + rec_rel) > 0 else 0.0
            
            avg_lag_rel = 0.0
            for (dc, ap) in assignments_rel:
                dc_center = (dc[0] + dc[1]) / 2.0
                ap_center = (ap[0] + ap[1]) / 2.0
                avg_lag_rel += abs(dc_center - ap_center)
            avg_lag_rel = avg_lag_rel / len(assignments_rel) if len(assignments_rel) > 0 else np.nan
            
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
        
    results_df = pd.DataFrame(eval_results)
    results_df.to_csv(os.path.join(output_eval_dir, "cdrift_evaluation_details.csv"), index=False)
    
    # Calculate dataset summaries and overall metrics
    summary_rows = []
    datasets = list(results_df["dataset"].unique()) + ["Overall"]
    for ds in datasets:
        if ds == "Overall":
            ds_df = results_df
        else:
            ds_df = results_df[results_df["dataset"] == ds]
            
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
        
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_eval_dir, "cdrift_evaluation_summary.csv"), index=False)
    
    print("\n================ EVALUATION SUMMARY (200 LAG) ================")
    print(summary_df.to_string(index=False))

    # Calculate global metrics
    global_rows = []
    lags = [200, 0.01, 0.025, 0.05]
    for lag in lags:
        if lag == 200:
            tp_col, fp_col, fn_col = "tp_200", "fp_200", "fn_200"
        else:
            tp_col, fp_col, fn_col = f"tp_{lag}", f"fp_{lag}", f"fn_{lag}"
            
        total_tp = results_df[tp_col].sum()
        total_fp = results_df[fp_col].sum()
        total_fn = results_df[fn_col].sum()
        
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
            "n_logs": len(results_df)
        })
        
    global_df = pd.DataFrame(global_rows)
    global_df.to_csv(os.path.join(output_eval_dir, "global_metrics.csv"), index=False)
    print("\n================ GLOBAL METRICS ================")
    print(global_df.to_string(index=False))

if __name__ == "__main__":
    log_dir = cfg.CDRIFT_LOG_DIR
    gt_csv = cfg.CDRIFT_GT_CSV
    
    # Update config overrides
    cfg.recalculate_derived_configs(args.base_rep)

    if args.n_windows is not None:
        cfg.N_WINDOWS = args.n_windows
        print(f"Overriding N_WINDOWS = {cfg.N_WINDOWS}")
    
    base_dir = os.path.join(ROOT_DIR, f"outputs/{cfg.experiment_name}_latest")
        
    preprocessed_dir = cfg.CDRIFT_OUTPUT_DIR
    model_path = cfg.TRAINED_MODEL_PATH
    output_eval_dir = os.path.join(ROOT_DIR, f"outputs/{cfg.experiment_name}_latest", "cdrift_evaluation")
    
    if getattr(cfg, 'AUTO_RUN_PREPROCESSING', False):
        print("Step 1: Preprocessing logs...")
        preprocess_cdrift_logs(log_dir, preprocessed_dir)
    else:
        print(f"Step 1: Skipping preprocessing (AUTO_RUN_PREPROCESSING=False). Using existing data in {preprocessed_dir}")
    
    print("\nStep 2: Loading ground truth...")
    gt_map = load_ground_truth(gt_csv)
    
    print("\nStep 3: Running prediction and evaluation...")
    run_prediction_and_evaluation(preprocessed_dir, model_path, gt_map, output_eval_dir)
    
    print(f"\nAll done! Results saved to {output_eval_dir}")
