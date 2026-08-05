import os
import sys
import json
import types
import pandas as pd
import tensorflow as tf

# Setup paths and module aliases for armature_light compatibility
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
obj_detection_dir = os.path.join(repo_root, "approaches", "object_detection")

if obj_detection_dir not in sys.path:
    sys.path.insert(0, obj_detection_dir)

import utils.armature_light.models as arm_models
src_mod = types.ModuleType('src')
arm_mod = types.ModuleType('armature_light')
src_mod.armature_light = arm_mod
arm_mod.models = arm_models
sys.modules['src'] = src_mod
sys.modules['src.armature_light'] = arm_mod
sys.modules['src.armature_light.models'] = arm_models

import utils.armature_light.discovery as arm_disc
arm_mod.discovery = arm_disc

import utils.config as cfg
import utils.utilities as utils
import utils.prediction as pred
import calculate_cdlg_evaluation_results as eval_cdlg


def run_prediction_and_evaluation():
    model_path = os.path.join(repo_root, "data", "output", "20240922-233643_winsim_sgd_model_4d_v1")
    base_custom_eval_dir = os.path.join(repo_root, "data", "custom_eval")

    subdirs = [d for d in os.listdir(base_custom_eval_dir) 
               if os.path.isdir(os.path.join(base_custom_eval_dir, d)) and not d.startswith(".")]

    subdirs = sorted(subdirs)

    print(f"Loaded trained model from: {model_path}")
    print(f"Found {len(subdirs)} custom_eval datasets: {subdirs}")

    # Load model once
    model = tf.saved_model.load(model_path)

    representations = ["dfg", "dfg_io", "arm"]
    summary_results = []

    for rep in representations:
        cfg.BASE_REPRESENTATION = rep
        print(f"\n=======================================================")
        print(f"  RUNNING EVALUATION FOR BASE REPRESENTATION: {rep.upper()}")
        print(f"=======================================================")

        for subdir in subdirs:
            log_dir = os.path.join(base_custom_eval_dir, subdir)
            drift_info_path = os.path.join(log_dir, "drift_info.csv")

            if not os.path.exists(drift_info_path):
                print(f"Skipping {subdir}: drift_info.csv not found.")
                continue

            output_dir = os.path.join(repo_root, "outputs", f"custom_eval_predictions_{rep}", subdir)
            winsim_img_dir = os.path.join(output_dir, "winsim_images")
            if os.path.exists(winsim_img_dir):
                import shutil
                shutil.rmtree(winsim_img_dir)
            for jf in ["window_info.json", "date_info.json", "number_of_traces.json", "prediction_results.csv"]:
                jp = os.path.join(output_dir, jf)
                if os.path.exists(jp):
                    os.remove(jp)
            os.makedirs(output_dir, exist_ok=True)

            print(f"\n--- Processing dataset '{subdir}' ({rep.upper()}) ---")

            # 1. Run Preprocessing / Prediction Pipeline
            image_dir = pred.prediction_pipeline(
                log_dir=log_dir,
                encoding_type="winsim",
                output_path=output_dir,
                n_windows=200
            )

            # Generate number_of_traces.json if not present
            num_traces_path = os.path.join(output_dir, "number_of_traces.json")
            if not os.path.exists(num_traces_path):
                log_files = utils.get_event_log_paths(log_dir)
                num_traces_dict = {}
                for log_name, log_p in log_files.items():
                    r_name = os.path.splitext(log_name)[0]
                    e_log = utils.import_event_log(log_p, log_name)
                    num_traces_dict[r_name] = len(e_log)
                    num_traces_dict[log_name] = len(e_log)
                with open(num_traces_path, "w", encoding="utf-8") as f:
                    json.dump(num_traces_dict, f, indent=2)

            # 2. Run Object Detection Prediction
            pred.predict(
                image_dir=image_dir,
                output_path=output_dir,
                model=model,
                encoding_type="winsim",
                n_windows=200
            )

            pred_results_csv = os.path.join(output_dir, "prediction_results.csv")
            total_metrics_file = os.path.join(output_dir, "total_metrics.txt")

            # 3. Calculate Ground Truth Evaluation Metrics
            try:
                eval_cdlg.evaluate(
                    prediction_results_path=pred_results_csv,
                    drift_info_path=drift_info_path,
                    number_of_traces_path=num_traces_path,
                    output_dir=output_dir,
                    lag=0.05,
                    confidence_threshold=0.5
                )

                # Parse metrics from total_metrics.txt
                prec, rec, f1 = 0.0, 0.0, 0.0
                if os.path.exists(total_metrics_file):
                    with open(total_metrics_file, "r") as f:
                        for line in f:
                            if line.startswith("Precision:"):
                                prec = float(line.split(":")[1].strip())
                            elif line.startswith("Recall:"):
                                rec = float(line.split(":")[1].strip())
                            elif line.startswith("F1:"):
                                f1 = float(line.split(":")[1].strip())

                summary_results.append({
                    "Representation": rep.upper(),
                    "Dataset": subdir,
                    "Precision": prec,
                    "Recall": rec,
                    "F1": f1,
                    "OutputDir": output_dir
                })

            except Exception as e:
                print(f"Error calculating metrics for {subdir} ({rep}): {e}")

    # Display overall summary table
    summary_df = pd.DataFrame(summary_results)
    print("\n\n=======================================================")
    print("      CUSTOM_EVALUATION COMPREHENSIVE SUMMARY          ")
    print("=======================================================")
    print(summary_df.to_string(index=False))

    summary_csv_path = os.path.join(repo_root, "outputs", "custom_eval_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSummary table saved to: {summary_csv_path}")


if __name__ == "__main__":
    run_prediction_and_evaluation()
