import os
from tqdm import tqdm
import pytz
import datetime as dt
import subprocess
from typing import Tuple
import re
import pandas as pd
import approaches.object_detection.utils.config as cfg
import approaches.object_detection.utils.evaluation as eval

def get_timestamp() -> str:
    """Get timestamp for current time.

    Returns:
        str: Timestamp as string in format "%Y%m%d-%H%M%S"
    """
    europe = pytz.timezone("Europe/Berlin")
    timestamp = dt.datetime.now(europe).strftime("%Y%m%d-%H%M%S")
    return timestamp


def get_event_log_paths(dir: str) -> dict:
    """Get event logs for directory.

    Args:
        dir (str): Log directory

    Returns:
        dict: Dictionary, containing names and paths of event logs
    """
    list_of_files = {}
    for dir_path, dir_names, filenames in os.walk(dir):
        for filename in filenames:
            if filename.endswith('.xes'):
                list_of_files[filename] = dir_path

    assert len(list_of_files) > 0, f"{dir} is empty"

    return list_of_files

def start_pro_drift_evaluation(window, out_path):
    """Handle for ProDrift evaluation.
    """
    files = get_event_log_paths(cfg.DEFAULT_LOG_DIR)

    for name, path in tqdm(files.items(), desc="Generate Predictions with ProDrift",
                           unit="Event Log"):
        log_path = os.path.join(path, name)
        outs, errs = call_pro_drift(log_path, cfg.PRODRIFT_DIR, window)
        with open(out_path, "a") as f:
            f.write(outs.decode())
            f.write("\n")


def call_pro_drift(log_path: str, pro_drift_dir: str,
                   window_size: int) -> Tuple[bytes, bytes]:
    """Call ProDrift for evaluation purposes.

    Args:
        log_path (str): Logs to evaluate
        pro_drift_dir (str): Directory of ProDrift distribution
        window_size (int): Window size for ProDrift

    Returns:
        Tuple[bytes, bytes]: Process output and error log
    """
    env = dict(os.environ)
    env['JAVA_OPTS'] = 'foo'

    cmd = f"java -jar ProDrift2.5.jar -fp {log_path} -ddm runs -ws {window_size} -gradual"
    #cmd = f"java -jar ProDrift2.5.jar -fp {log_path} -ddm runs -gradual"

    p = subprocess.Popen(cmd,
                         stdout=subprocess.PIPE,
                         shell=True,
                         cwd=pro_drift_dir,
                         env=env)
    try:
        outs, errs = p.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        p.kill()
        outs, errs = p.communicate()

    return outs, errs






def transform_txt_to_csv(file_location, file_name):

    path_to_log = os.path.join(file_location, file_name)

    # load txt file
    with open(path_to_log, 'r') as file:
        lines = file.readlines()

    # Initialize empty lists to store data
    log_data = {}

    # Iterate through each line
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace
        if line.startswith("log_"):  # Check if the line contains log name
            log_name = line.split()[0]  # Extract log name
            if log_name not in log_data:
                log_data[log_name] = {"drift_types": [], "change_trace_idx": []}
        elif "Sudden drift detected" in line:  # Check if the line contains sudden drift
            drift_type = "sudden"
            change_point = int(line.split(": ")[-1].split()[0])  # Extract change point
            change_point_tuple = tuple([change_point, change_point])
            log_data[log_name]["drift_types"].append(drift_type)
            log_data[log_name]["change_trace_idx"].append(change_point_tuple)
        elif "Gradual drift detected" in line:  # Check if the line contains gradual drift
            drift_type = "gradual"
            #pattern = r'\((\d+)\)\s(.*?)\sdrift detected at trace: (\d+) \((.*?)\) after reading \d+ traces\.'
            #re.findall(pattern, line)
            change_points = re.findall(r'trace: (\d+)', line)
            change_point_tuple = tuple(map(int, change_points))
            log_data[log_name]["drift_types"].append(drift_type)
            log_data[log_name]["change_trace_idx"].append(change_point_tuple)

    # Convert the dictionary into a list of dictionaries
    data_list = []
    for log_name, log_info in log_data.items():
        if len(log_info["drift_types"]) == 0:
            data_list.append({
                "log_name": log_name,
                "drift_types": "",
                "change_trace_idx": ""})
        else:
            for i in range(len(log_info["drift_types"])):
                data_list.append({
                    "log_name": log_name,
                    "drift_types": log_info["drift_types"][i],
                    "change_trace_idx": log_info["change_trace_idx"][i]
                })

    # Create a DataFrame
    df = pd.DataFrame(data_list)

    # group by log_name
    df_grouped = df.groupby(["log_name"]).agg({
        "drift_types": lambda x: ', '.join(x),
        "change_trace_idx": lambda x: list(x)})

    # Replace values in the column 'change_trace_idx' with "" if this is present in the column 'drift_types'
    df_grouped.loc[df_grouped['drift_types'] == "", 'change_trace_idx'] = ""

    # Print/save the DataFrame
    file_name_csv = file_name.split(".")[0] + ".csv"
    path_csv_file = os.path.join(file_location, file_name_csv)
    df_grouped.reset_index(inplace=True)
    df_grouped.to_csv(path_csv_file, index=False)

    return df_grouped, path_csv_file


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # STEP 1: run ProDrift tool
#    window_size = 300
#    out_path = os.path.abspath(os.path.join("data", "output", "ProDrift", f"{get_timestamp()}_prodrift_results_{window_size}.txt"))
#    out_path = os.path.abspath(os.path.join("data", "output", "ProDrift", f"{get_timestamp()}_prodrift_results_adwin.txt"))
#    start_pro_drift_evaluation(window_size, out_path)

    # Step 2.1: Transformations txt to csv file for results from ProDrift
#     window_sizes = ['window_50','window_100', 'window_150','window_200', 'window_250', 'window_300','window_adwin']
    window_sizes = ['window_300']

    for window in window_sizes:
        size = window.split("_")[1]
        file_name = f'prodrift_results_{size}.txt'
        file_location = f'/.../scdd/data/output/ProDrift/{window}'

        df, path_csv_file = transform_txt_to_csv(file_location, file_name)

        # Step 2.2: evaluate the results
        eval.evaluate_pro_drift_results(results_file_path=path_csv_file, data_dir=cfg.TEST_IMAGE_DATA_DIR)

