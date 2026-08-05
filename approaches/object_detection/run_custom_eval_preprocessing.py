import os
import sys
import json
import shutil
import types
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 1. Setup paths and module aliases for armature_light compatibility BEFORE importing discovery
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
obj_detection_dir = os.path.join(repo_root, "approaches", "object_detection")

if obj_detection_dir not in sys.path:
    sys.path.insert(0, obj_detection_dir)

# Import models first
import utils.armature_light.models as arm_models

# Populate sys.modules for 'src' and 'src.armature_light'
src_mod = types.ModuleType('src')
arm_mod = types.ModuleType('armature_light')
src_mod.armature_light = arm_mod
arm_mod.models = arm_models
sys.modules['src'] = src_mod
sys.modules['src.armature_light'] = arm_mod
sys.modules['src.armature_light.models'] = arm_models

# Now import discovery safely
import utils.armature_light.discovery as arm_disc
arm_mod.discovery = arm_disc

import utils.config as cfg
import utils.utilities as utils
import utils.preprocessing as pp


def preprocess_log_for_rep(log_path: str, output_dir: str, base_rep: str, n_windows: int = 200):
    cfg.BASE_REPRESENTATION = base_rep
    os.makedirs(output_dir, exist_ok=True)

    log_dir, log_filename = os.path.split(log_path)
    real_name = os.path.splitext(log_filename)[0]

    print(f"--- Preprocessing {log_filename} with {base_rep.upper()} (n_windows={n_windows}) ---")

    # Import log & filter complete events
    event_log = utils.import_event_log(path=log_dir, name=log_filename)
    filtered_log = utils.filter_complete_events(event_log)
    n_traces = utils.get_number_of_traces(filtered_log)

    # Compute windowed matrices & similarity matrix
    if base_rep == "arm":
        windowed_matrices, borders, window_info, date_info = pp.log_to_windowed_arm(filtered_log, n_windows)
        sim_matrix = pp.similarity_calculation_arm(windowed_matrices)
    else:
        windowed_matrices, borders, window_info, date_info = pp.log_to_windowed_dfg_count(filtered_log, n_windows)
        sim_matrix = pp.similarity_calculation(windowed_matrices)

    # Save as image using viridis (standard color mode)
    cm = plt.get_cmap('viridis')
    colored_image = cm(sim_matrix / 255.0 if sim_matrix.max() > 1 else sim_matrix)
    img = Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8))
    
    img_save_path = os.path.join(output_dir, f"{real_name}.jpg")
    img.save(img_save_path)
    print(f"  Saved image to: {img_save_path}")

    # Save metadata JSONs
    win_info_data = {}
    win_info_path = os.path.join(output_dir, "window_info.json")
    if os.path.exists(win_info_path):
        with open(win_info_path, "r", encoding="utf-8") as f:
            win_info_data = json.load(f)
    win_info_data[real_name] = window_info
    with open(win_info_path, "w", encoding="utf-8") as f:
        json.dump(win_info_data, f, indent=2)

    date_info_data = {}
    date_info_path = os.path.join(output_dir, "date_info.json")
    if os.path.exists(date_info_path):
        with open(date_info_path, "r", encoding="utf-8") as f:
            date_info_data = json.load(f)
    date_info_data[real_name] = date_info
    with open(date_info_path, "w", encoding="utf-8") as f:
        json.dump(date_info_data, f, indent=2)

    num_traces_data = {}
    num_traces_path = os.path.join(output_dir, "number_of_traces.json")
    if os.path.exists(num_traces_path):
        with open(num_traces_path, "r", encoding="utf-8") as f:
            num_traces_data = json.load(f)
    num_traces_data[real_name] = n_traces
    with open(num_traces_path, "w", encoding="utf-8") as f:
        json.dump(num_traces_data, f, indent=2)

    print(f"  Successfully finished {base_rep.upper()} preprocessing for {real_name}.")


def main():
    trace_dir = os.path.join(repo_root, "data", "custom_eval", "related_conditions_traces")
    target_log_path = os.path.join(trace_dir, "related_condition.xes")
    source_log_path = os.path.join(repo_root, "data", "custom_eval", "related_conditions", "process_drift_log.xes")

    os.makedirs(trace_dir, exist_ok=True)
    if not os.path.exists(target_log_path) and os.path.exists(source_log_path):
        shutil.copy(source_log_path, target_log_path)

    # Get ALL .xes files in related_conditions_traces
    logs_to_process = sorted([
        os.path.join(trace_dir, f) for f in os.listdir(trace_dir) if f.endswith(".xes")
    ])

    print(f"Found {len(logs_to_process)} log(s) to process in {trace_dir}:")
    for lp in logs_to_process:
        print(f" - {os.path.basename(lp)}")

    out_base_custom = os.path.join(repo_root, "outputs", "test", "custom_eval")

    representations = ["dfg", "dfg_io", "arm"]

    for rep in representations:
        rep_out_custom = os.path.join(out_base_custom, rep)

        for log_p in logs_to_process:
            preprocess_log_for_rep(log_p, rep_out_custom, base_rep=rep)
            preprocess_log_for_rep(log_p, out_base_custom, base_rep=rep)

    print("\nAll preprocessing for all logs completed successfully!")


if __name__ == "__main__":
    main()
