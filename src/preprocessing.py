import os
import json
import numpy as np
import pandas as pd
import src.utilities as utils
import src.config as cfg
from pm4py import discover_dfg_typed
import multiprocessing
from functools import partial

from numpy import linalg as LA
from scipy import spatial
from scipy.stats import wasserstein_distance
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.log.obj import EventLog
from typing import Tuple, Dict, List
from tqdm import tqdm
from src.armature_light.models import Trace, Event
from src.armature_light import discovery


def extract_weights_to_arrays(matrix_obj: Dict) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Converts heavy matrix objects into a dictionary of lean NumPy arrays.
    """
    extracted = {}
    for pair, weight in matrix_obj.items():
        # Combine temporal (4) and existential (4) into one vector
        t = weight.temporal
        e = weight.existential

        arr = np.array([
            t.direct, t.direct_backward, t.true_eventual, t.true_eventual_backward,
            e.count_both, e.count_only_a, e.count_only_b, e.count_neither
        ], dtype=np.float32)

        extracted[pair] = arr
    return extracted

def build_feature_matrix(idx1: int, idx2: int, windows_data: List[Dict]):
    """
    Transforms a pair of sparse dictionaries into aligned dense 2D matrices.
    """
    m1 = windows_data[idx1]
    m2 = windows_data[idx2]

    # 1. Get the union of keys and lock the order
    # Using the | operator is a fast way to union dictionary keys in Python 3
    all_keys = list(m1.keys() | m2.keys())
    n_keys = len(all_keys)

    # 2. Pre-allocate 2D dense matrices of shape (n_keys, 8)
    mat1 = np.zeros((n_keys, 8), dtype=np.float32)
    mat2 = np.zeros((n_keys, 8), dtype=np.float32)

    # 3. Populate the matrices
    for i, k in enumerate(all_keys):
        if k in m1:
            mat1[i] = m1[k]
        if k in m2:
            mat2[i] = m2[k]

    return mat1, mat2



def compute_single_pair(pair_indices: Tuple[int, int], windows_data: List[Dict]):
    """Worker function for parallel processing."""
    idx1, idx2 = pair_indices
    m1, m2 = build_feature_matrix(idx1, idx2, windows_data)
    dist_value = calc_distance_norm(m1, m2, option=cfg.DISTANCE_MEASURE)
    return idx1, idx2, dist_value

def process_log(name_path):
    name, path = name_path 
    # Try to extract integer index from log name (e.g. log_356_1726735302.xes -> 356)
    # This format is used by the CDLG dataset generator.
    try:
        parts = str(name).split("_")
        if len(parts) > 1 and parts[0] == "log":
            drift_number = int(parts[1])
        else:
            # For non-CDLG logs (e.g. CDrift), use a deterministic hash of the
            # full filename to avoid collisions. The old regex approach picked
            # the first digit sequence, causing logs like
            # sudden_trace_noise5_1000_pl.xes and sudden_trace_noise5_1000_cb.xes
            # to both map to image "5" and overwrite each other.
            drift_number = abs(hash(name)) % 10**8
    except (ValueError, IndexError):
        drift_number = abs(hash(name)) % 10**8

    # load event log
    event_log = utils.import_event_log(path=path, name=name)

    # if the log contains incomplete traces, the log is filtered
    filtered_log = utils.filter_complete_events(event_log)

    if cfg.BASE_REPRESENTATION in ["dfg", "dfg_io"]:
        windowed_matrices, borders, window_information, log_date_info = \
            log_to_windowed_dfg_count(filtered_log, add_io=(cfg.BASE_REPRESENTATION == "dfg_io"))
    else:
        windowed_matrices, borders, window_information, log_date_info = \
            log_to_windowed_arm(filtered_log)

    number_of_traces = utils.get_number_of_traces(filtered_log)
    # get similarity matrix
    sim_matrix = similarity_calculation(windowed_matrices)

    # save matrix as image
    utils.matrix_to_img(matrix=sim_matrix,
                        number=drift_number,
                        exp_path=cfg.DEFAULT_DATA_DIR,
                        mode=cfg.COLOR)
    print(f"Log finsihed: {name}")
    return name, drift_number, borders, window_information, log_date_info, number_of_traces

def process_generic_directory(input_dir: str, output_dir: str, create_tfrecords: bool = False):
    """
    A generic entry point to run preprocessing on any arbitrary folder of .xes logs.
    """
    os.makedirs(output_dir, exist_ok=True)
    cfg.DEFAULT_LOG_DIR = input_dir
    cfg.DEFAULT_DATA_DIR = output_dir
    cfg.TFR_RECORDS_DIR = output_dir
    cfg.AUTOMATE_TFR_SCRIPT = create_tfrecords
    cfg.OUTPUT_PREFIX = "model_4d_cd"
    
    winsim_pipeline()

def winsim_pipeline():
    """Main function for preprocessing the event logs for the WINSIM approach.
    """
    # create experiment folder structure
    cfg.DEFAULT_DATA_DIR = utils.create_winsim_experiment(cfg.DEFAULT_DATA_DIR)

    # get all paths and file names of event logs
    log_files = utils.get_event_log_paths(cfg.DEFAULT_LOG_DIR)

    try:
        drift_info = utils.extract_drift_information(cfg.DEFAULT_LOG_DIR)
    except FileNotFoundError:
        drift_info = None

    log_matching = {}
    window_info = {}
    date_info = {}
    number_of_traces = {}

    # Load existing metadata if skipping is possible
    window_info_path = os.path.join(cfg.DEFAULT_DATA_DIR, "window_info.json")
    if os.path.exists(window_info_path):
        try:
            with open(window_info_path, "r", encoding='utf-8') as f:
                window_info = json.load(f)
            with open(os.path.join(cfg.DEFAULT_DATA_DIR, "date_info.json"), "r", encoding='utf-8') as f:
                date_info = json.load(f)
            with open(os.path.join(cfg.DEFAULT_DATA_DIR, "number_of_traces.json"), "r", encoding='utf-8') as f:
                number_of_traces = json.load(f)
            
            # Load drift info if exists
            drift_csv = os.path.join(cfg.DEFAULT_DATA_DIR, "drift_info.csv")
            if os.path.exists(drift_csv):
                drift_info = pd.read_csv(drift_csv, sep=";")
                
            log_matching_csv = os.path.join(cfg.DEFAULT_DATA_DIR, "log_matching.csv")
            if os.path.exists(log_matching_csv):
                log_matching_df = pd.read_csv(log_matching_csv)
                # Need to support both index_label='log_name' and unnamed index cases
                if "log_name" in log_matching_df.columns:
                    log_matching_df = log_matching_df.set_index("log_name")
                elif log_matching_df.columns[0] == "Unnamed: 0":
                    log_matching_df = log_matching_df.set_index("Unnamed: 0")
                log_matching = log_matching_df["image_id"].to_dict()
                
            # Filter log_files to skip already processed ones
            ext = ".png" if getattr(cfg, 'COLOR', 'grayscale') == "grayscale" else ".jpg"
            logs_to_process = {}
            for name, path in log_files.items():
                img_name = log_matching.get(name)
                if img_name is None:
                    # Fallback to the same extraction process as process_log
                    try:
                        parts = str(name).split("_")
                        if len(parts) > 1 and parts[0] == "log":
                            img_name = int(parts[1])
                        else:
                            img_name = abs(hash(name)) % 10**8
                    except:
                        img_name = abs(hash(name)) % 10**8

                img_path = os.path.join(cfg.DEFAULT_DATA_DIR, f"{img_name}{ext}")
                if not (os.path.exists(img_path) and name in window_info):
                    logs_to_process[name] = path
            
            log_files = logs_to_process
            print(f"Found {len(log_files)} logs left to process (skipping already processed).")
        except Exception as e:
            print(f"Could not load existing metadata for skipping: {e}")

    num_cores = getattr(cfg, 'N_CORES_WINSIM', 4)

    with multiprocessing.Pool(num_cores) as pool:
        results = list(tqdm(pool.imap(process_log, log_files.items()),
                            desc="Preprocessing Event Logs", unit=" Event Log"))

    for result in tqdm(results, desc="Finishing preprocessing of Event Logs"):
        name, drift_number, borders, window_information, log_date_info, n_of_traces = result
        if drift_info is not None:
            drift_info = utils.update_trace_indices(drift_info, name, borders)
        window_info[name] = window_information
        log_matching[name] = drift_number
        date_info[name] = log_date_info
        number_of_traces[name] = n_of_traces


    if drift_info is not None:
        drift_info.to_csv(os.path.join(cfg.DEFAULT_DATA_DIR, "drift_info.csv"), sep=";")

    log_matching_df = pd.DataFrame.from_dict(log_matching,
                                             orient="index",
                                             columns=["image_id"])
    # Save without index to avoid issues with dict conversion later
    log_matching_df.to_csv(os.path.join(cfg.DEFAULT_DATA_DIR, "log_matching.csv"), index_label="log_name")

    window_info_path = os.path.join(cfg.DEFAULT_DATA_DIR, "window_info.json")
    with open(window_info_path, "w", encoding='utf-8') as file:
        json.dump(window_info, file)

    date_info_path = os.path.join(cfg.DEFAULT_DATA_DIR, "date_info.json")
    with open(date_info_path, "w", encoding='utf-8') as file:
        json.dump(date_info, file)

    number_of_traces_path = os.path.join(cfg.DEFAULT_DATA_DIR, "number_of_traces.json")
    with open(number_of_traces_path, "w", encoding='utf-8') as file:
        json.dump(number_of_traces, file)

    if drift_info is not None:
        utils.generate_annotations(drift_info,
                                   dir=cfg.DEFAULT_DATA_DIR,
                                   log_matching=log_matching,
                                   log_names=log_files.keys())

    if cfg.AUTOMATE_TFR_SCRIPT:
        utils.start_tfr_script(repo_dir=cfg.TENSORFLOW_MODELS_DIR,
                               data_dir=cfg.DEFAULT_DATA_DIR,
                               tfr_dir=cfg.TFR_RECORDS_DIR,
                               prefix=cfg.OUTPUT_PREFIX)
 

def log_to_windowed_dfg_count(event_log: EventLog, add_io: bool = False) \
    -> Tuple[np.ndarray, list, dict, tuple]:
    """Convert event log to directly follows frequency counts for each activity.

    Args:
        event_log (EventLog): Event log

    Returns:
        Tuple[np.ndarray,list,dict,tuple]: Tuple, containing the dfg frequency counts, 
        border information of windows in form of trace indices, 
        window information with first trace and timestamp and
        date information with first and last timestamp of event log
    """
    # get dataframe
    event_log_df = log_converter.apply(
        event_log, variant=log_converter.Variants.TO_DATA_FRAME)
    event_log_df = dataframe_utils.convert_timestamp_columns_in_df(
        event_log_df, timest_format="ISO8601")

    if add_io:
        start_name = "Artificial_Start"
        end_name = "Artificial_End"

        # Preserve the exact original interleaving of the event log
        event_log_df = event_log_df.reset_index(drop=True)
        event_log_df['_order'] = event_log_df.index.astype(float)

        first_rows = event_log_df.groupby('case:concept:name').first().reset_index()
        last_rows = event_log_df.groupby('case:concept:name').last().reset_index()

        start_events = first_rows.copy()
        start_events['concept:name'] = start_name
        start_events['time:timestamp'] = start_events['time:timestamp'] - pd.Timedelta(seconds=1)
        start_events['_order'] = start_events['_order'] - 0.1

        end_events = last_rows.copy()
        end_events['concept:name'] = end_name
        end_events['time:timestamp'] = end_events['time:timestamp'] + pd.Timedelta(seconds=1)
        end_events['_order'] = end_events['_order'] + 0.1

        event_log_df = pd.concat([event_log_df, start_events, end_events], ignore_index=True)
        event_log_df = event_log_df.sort_values('_order').drop(columns=['_order']).reset_index(drop=True)

    min_date = np.min(event_log_df["time:timestamp"])
    max_date = np.max(event_log_df["time:timestamp"])
    date_info = (utils.datetime_2_str(min_date), 
                 utils.datetime_2_str(max_date))
    
    # get unique event names
    act_names = np.unique(event_log_df["concept:name"])

    # get unique trace names
    unique_traces = pd.unique(event_log_df["case:concept:name"])
    
    # get window size based on number of windows and event log size
    n_windows = cfg.N_WINDOWS
    window_size = len(event_log) // n_windows
    if window_size == 0:
        raise ValueError(
            f"The log length ({len(event_log)}) is too small to be divided by n_windows ({n_windows}).")
    
    # initialize helper variables
    freq_count = 0
    left_boundary = 0
    right_boundary = window_size
    borders = []
    dfg_graphs = []

    window_information = {}

    # iterate through windows
    for i in range(1, n_windows + 1):
        dfg_matrix_df = pd.DataFrame(0, columns=act_names, index=act_names)

        # get all trace names that are in selected window, traces are sorted by timestamp
        if i < n_windows:
            w_unique_traces = unique_traces[left_boundary:right_boundary]
        # at last window fill until the end of the list
        else:
            w_unique_traces = unique_traces[left_boundary:]
            right_boundary = len(unique_traces) - 1 

        # search all events for given traces
        log_window = event_log_df[event_log_df["case:concept:name"]
                                  .isin(w_unique_traces)]

        # get dfg graph for window
        graph, _, _ = discover_dfg_typed(log_window)

        # transform dfg graph into dfg matrix
        for relation, freq in graph.items():
            rel_a, rel_b = relation
            dfg_matrix_df.at[rel_a, rel_b] = freq

            freq_count += freq

        dfg_graphs.append(dfg_matrix_df)

        borders.append((left_boundary, right_boundary))

        left_boundary = right_boundary
        right_boundary += window_size

        # get id of first trace in window - for evaluation
        first_trace = w_unique_traces[0]
        first_timestamp = min(
            event_log_df.loc[event_log_df["case:concept:name"] == first_trace,
                             "time:timestamp"])
        first_timestamp = first_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        window_information[i] = (first_trace, first_timestamp)

    # compare dfg frequencies of all windows with dfg frequencies of complete log
    # ensures that there are no missing relations
    graph, _, _ = discover_dfg_typed(event_log_df)
    total_freq = sum(graph.values())
    assert total_freq == freq_count, (
        "Missing directly follow relations.\n"
        f"Number of relations in whole event log: {total_freq}.\n"
        f"Number of relations in all windows: {freq_count}"
    )
    
    return np.array(dfg_graphs), borders, window_information, date_info


def log_to_windowed_arm(event_log: EventLog) \
    -> Tuple[List[Dict], list, dict, tuple]:
    """Convert event log to Activity Relationship Matrix (ARM) weights for each window.

    Args:
        event_log (EventLog): Event log

    Returns:
        Tuple[List[Dict], list, dict, tuple]: Tuple, containing the ARM weights as a list of dictionaries, 
        border information of windows in form of trace indices, 
        window information with first trace and timestamp and
        date information with first and last timestamp of event log
    """

    # convert event log to pandas dataframe
    event_log_df = log_converter.apply(
        event_log, variant=log_converter.Variants.TO_DATA_FRAME)
    event_log_df = dataframe_utils.convert_timestamp_columns_in_df(
        event_log_df, timest_format="ISO8601")

    min_date = np.min(event_log_df["time:timestamp"])
    max_date = np.max(event_log_df["time:timestamp"])
    date_info = (utils.datetime_2_str(min_date), 
                 utils.datetime_2_str(max_date))

    # get unique trace names
    unique_traces = pd.unique(event_log_df["case:concept:name"])

    # get window size based on number of windows and event log size
    n_windows = cfg.N_WINDOWS
    window_size = len(event_log) // n_windows
    if window_size == 0:
        raise ValueError(
            f"The log length ({len(event_log)}) is too small to be divided by n_windows ({n_windows}).")

    # initialize helper variables
    left_boundary = 0
    right_boundary = window_size
    borders = []
    arm_graphs = []

    window_information = {}

    # iterate through windows
    for i in range(1, n_windows + 1):
        if i < n_windows:
            w_unique_traces = unique_traces[left_boundary:right_boundary]
        else:
            w_unique_traces = unique_traces[left_boundary:]
            right_boundary = len(unique_traces) - 1 

        # search all events for given traces
        log_window = event_log_df[event_log_df["case:concept:name"]
                                  .isin(w_unique_traces)]

        # convert log_window (DataFrame) to armature-light traces
        traces = []
        for case_id, group in log_window.groupby("case:concept:name", sort=False):
            events = [Event(activity=row["concept:name"], timestamp=row["time:timestamp"]) 
                      for _, row in group.iterrows()]
            traces.append(Trace(case_id=str(case_id), events=events))

        # compute ARM weights
        raw_matrix = discovery.compute_weights(traces)
        extracted_matrix = extract_weights_to_arrays(raw_matrix)

        arm_graphs.append(extracted_matrix)
        borders.append((left_boundary, right_boundary))

        left_boundary = right_boundary
        right_boundary += window_size

        # get id of first trace in window - for evaluation
        first_trace = w_unique_traces[0]
        first_timestamp = min(
            event_log_df.loc[event_log_df["case:concept:name"] == first_trace,
                             "time:timestamp"])
        first_timestamp = first_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        window_information[i] = (first_trace, first_timestamp)

    return arm_graphs, borders, window_information, date_info


def similarity_calculation(windowed_matrices) -> np.ndarray:
    """
    Compute similarity values for each element in the windowed matrices.
    Supports both DFG and ARM representations.
    """
    n = len(windowed_matrices)
    sim_matrix = np.zeros((n, n))

    if cfg.BASE_REPRESENTATION in ["dfg", "dfg_io"]:
        for i in range(n):
            for j in range(i + 1, n):
                val = calc_distance_norm(windowed_matrices[i], windowed_matrices[j], cfg.DISTANCE_MEASURE)
                sim_matrix[i, j] = val
                sim_matrix[j, i] = val
    else:
        # Sequential Loop for ARM
        for i in range(n):
            for j in range(i + 1, n):
                _, _, val = compute_single_pair((i, j), windowed_matrices)
                sim_matrix[i, j] = val
                sim_matrix[j, i] = val

    # Check diagonal of similarity matrix
    assert np.sum(np.diagonal(sim_matrix)) == 0, "The diagonal must be zero."

    # Normalize and invert (Lower diff = Higher similarity)
    if np.max(sim_matrix) > 0:
        sim_matrix = 1.0 - (sim_matrix / np.max(sim_matrix))

    # Transform to 0-255 for image representation
    img_matrix = np.uint8(sim_matrix * 255)

    return img_matrix

def calc_distance_norm(matrix_1: np.ndarray, matrix_2: np.ndarray, 
                       option: str) -> float:
    """Calculate distance/similarity value using the given directly follows matrices.

    Args:
        matrix_1 (np.ndarray): Directly follows matrix
        matrix_2 (np.ndarray): Directly follows matrix
        option (str): Indicates which measure to use for calculation

    Returns:
        float: Similarity value
    """
    diff = matrix_1 - matrix_2
    if option == "fro":
        # Frobenius norm
        dist_value = LA.norm(diff, "fro")
    elif option == "nuc":
        # nuclear norm
        dist_value = LA.norm(diff, "nuc")
    elif option == "inf":
        # max norm
        dist_value = LA.norm(diff, np.inf)
    elif option == "l2":
        # L2 norm
        dist_value = LA.norm(diff, 2)
    elif option == "cos":
        # cosine distance
        dist_value = spatial.distance.cosine(
            matrix_1.ravel(), matrix_2.ravel())
    elif option == "earth":
        # earth mover distance == wasserstein distance
        dist_value = 0

        # compute histograms and wasserstein distance for each activity
        for i in range(0, len(matrix_1)):
            matrix_1_hist, _ = np.histogram(
                matrix_1[i, :], bins=np.arange(-0.5, len(matrix_1)), density=True)
            matrix_2_hist, _ = np.histogram(
                matrix_2[i, :], bins=np.arange(-0.5, len(matrix_2)), density=True)
            dist_value += wasserstein_distance(matrix_1_hist, matrix_2_hist)

    if np.isnan(dist_value):
        dist_value = LA.norm(diff, "fro")

    return dist_value


if __name__ == "__main__":
    
    winsim_pipeline(cfg.N_WINDOWS)
