import os
import json
import numpy as np
import pandas as pd
import utils.utilities as utils
import utils.config as cfg
import utils.vdd_helper as vdd_helper
import utils.vdd_data_analysis as vdd
import multiprocessing
from functools import partial

from pm4py import discover_dfg_typed
from numpy import linalg as LA
from scipy import spatial
from scipy.stats import wasserstein_distance
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.log.obj import EventLog
from typing import Tuple, Dict, List
from tqdm import tqdm

from utils.armature_light.models import Trace, Event
from utils.armature_light import discovery

def extract_weights_to_arrays(matrix_obj: Dict) -> Dict[Tuple[str, str], np.ndarray]:
    extracted = {}
    for pair, weight in matrix_obj.items():
        t = weight.temporal
        e = weight.existential
        arr = np.array([
            t.direct, t.direct_backward, t.true_eventual, t.true_eventual_backward,
            e.count_both, e.count_only_a, e.count_only_b, e.count_neither
        ], dtype=np.float32)
        extracted[pair] = arr
    return extracted

def build_feature_matrix(idx1: int, idx2: int, windows_data: List[Dict]):
    m1 = windows_data[idx1]
    m2 = windows_data[idx2]
    all_keys = list(m1.keys() | m2.keys())
    n_keys = len(all_keys)
    mat1 = np.zeros((n_keys, 8), dtype=np.float32)
    mat2 = np.zeros((n_keys, 8), dtype=np.float32)
    for i, k in enumerate(all_keys):
        if k in m1:
            mat1[i] = m1[k]
        if k in m2:
            mat2[i] = m2[k]
    return mat1, mat2

def compute_single_pair(pair_indices: Tuple[int, int], windows_data: List[Dict]):
    idx1, idx2 = pair_indices
    m1, m2 = build_feature_matrix(idx1, idx2, windows_data)
    dist_value = calc_distance_norm(m1, m2, option=cfg.DISTANCE_MEASURE)
    return idx1, idx2, dist_value

def log_to_windowed_arm(event_log: EventLog, n_windows: int) -> Tuple[List[Dict], list, dict, tuple]:
    event_log_df = log_converter.apply(event_log, variant=log_converter.Variants.TO_DATA_FRAME)
    event_log_df = dataframe_utils.convert_timestamp_columns_in_df(event_log_df, timest_format='ISO8601')
    min_date = np.min(event_log_df['time:timestamp'])
    max_date = np.max(event_log_df['time:timestamp'])
    date_info = (utils.datetime_2_str(min_date), utils.datetime_2_str(max_date))
    unique_traces = pd.unique(event_log_df['case:concept:name'])
    window_size = len(event_log) // n_windows
    if window_size == 0:
        raise ValueError(f'The log length ({len(event_log)}) is too small.')
    left_boundary = 0
    right_boundary = window_size
    borders = []
    arm_graphs = []
    window_information = {}
    for i in range(1, n_windows + 1):
        if i < n_windows:
            w_unique_traces = unique_traces[left_boundary:right_boundary]
        else:
            w_unique_traces = unique_traces[left_boundary:]
            right_boundary = len(unique_traces) - 1 
        log_window = event_log_df[event_log_df['case:concept:name'].isin(w_unique_traces)]
        traces = []
        for case_id, group in log_window.groupby('case:concept:name', sort=False):
            events = [Event(activity=row['concept:name'], timestamp=row['time:timestamp']) for _, row in group.iterrows()]
            traces.append(Trace(case_id=str(case_id), events=events))
        raw_matrix = discovery.compute_weights(traces)
        extracted_matrix = extract_weights_to_arrays(raw_matrix)
        arm_graphs.append(extracted_matrix)
        borders.append((left_boundary, right_boundary))
        left_boundary = right_boundary
        right_boundary += window_size
        first_trace = w_unique_traces[0]
        first_timestamp = min(event_log_df.loc[event_log_df['case:concept:name'] == first_trace, 'time:timestamp'])
        first_timestamp = first_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        window_information[i] = (first_trace, first_timestamp)
    return arm_graphs, borders, window_information, date_info

def similarity_calculation_arm(windowed_matrices) -> np.ndarray:
    n = len(windowed_matrices)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            _, _, val = compute_single_pair((i, j), windowed_matrices)
            sim_matrix[i, j] = val
            sim_matrix[j, i] = val
    assert np.sum(np.diagonal(sim_matrix)) == 0, 'The diagonal must be zero.'
    if np.max(sim_matrix) > 0:
        sim_matrix = 1.0 - (sim_matrix / np.max(sim_matrix))
    img_matrix = np.uint8(sim_matrix * 255)
    return img_matrix



def process_log(name_path, n_windows):
    name, path = name_path 
    if getattr(cfg, 'BASE_REPRESENTATION', 'dfg') == 'arm':
        try:
            parts = str(name).split('_')
            if len(parts) > 1 and parts[0] == 'log':
                drift_number = int(parts[1])
            else:
                drift_number = abs(hash(name)) % 10**8
        except (ValueError, IndexError):
            drift_number = abs(hash(name)) % 10**8
    else:
        drift_number = int(name.split("_")[1])

    # load event log
    event_log = utils.import_event_log(path=path, name=name)

    # if the log contains incomplete traces, the log is filtered
    filtered_log = utils.filter_complete_events(event_log)

    if getattr(cfg, 'BASE_REPRESENTATION', 'dfg') == 'arm':
        windowed_matrices, borders, window_information, log_date_info = \
            log_to_windowed_arm(filtered_log, n_windows)
        number_of_traces = utils.get_number_of_traces(filtered_log)
        sim_matrix = similarity_calculation_arm(windowed_matrices)
    else:
        windowed_matrices, borders, window_information, log_date_info = \
            log_to_windowed_dfg_count(filtered_log, n_windows)
        number_of_traces = utils.get_number_of_traces(filtered_log)
        sim_matrix = similarity_calculation(windowed_matrices)

    # save matrix as image
    utils.matrix_to_img(matrix=sim_matrix,
                        number=drift_number,
                        exp_path=cfg.DEFAULT_DATA_DIR,
                        mode=cfg.COLOR)
    print(f"Log finsihed: {name}")
    return name, drift_number, borders, window_information, log_date_info, number_of_traces


def winsim_pipeline(n_windows=100):
    """Main function for preprocessing the event logs for the WINSIM approach.

    Args:
        n_windows (int, optional): Number of windows. Defaults to 100.
        p_mode (str, optional): Preprocessing mode. Defaults to "train".
    """
    # create experiment folder structure
    cfg.DEFAULT_DATA_DIR = utils.create_winsim_experiment(cfg.DEFAULT_DATA_DIR)

    # get all paths and file names of event logs
    log_files = utils.get_event_log_paths(cfg.DEFAULT_LOG_DIR)

    drift_info = utils.extract_drift_information(cfg.DEFAULT_LOG_DIR)

    log_matching = {}
    window_info = {}
    date_info = {}
    number_of_traces = {}

    num_cores = cfg.N_CORES_WINSIM

    with multiprocessing.Pool(num_cores) as pool:
        partial_process_log = partial(process_log, n_windows=n_windows)
        results = list(tqdm(pool.imap(partial_process_log, log_files.items()),
                            desc="Preprocessing Event Logs", unit=" Event Log"))

    for result in tqdm(results, desc="Finishing preprocessing of Event Logs"):
        name, drift_number, borders, window_information, log_date_info, n_of_traces = result
        drift_info = utils.update_trace_indices(drift_info, name, borders)
        window_info[name] = window_information
        log_matching[name] = drift_number
        date_info[name] = log_date_info
        number_of_traces[name] = n_of_traces


    drift_info.to_csv(os.path.join(cfg.DEFAULT_DATA_DIR, "drift_info.csv"))

    log_matching_df = pd.DataFrame.from_dict(log_matching,
                                             orient="index",
                                             columns=["image_id"])
    log_matching_df.to_csv(os.path.join(cfg.DEFAULT_DATA_DIR, "log_matching.csv"))

    window_info_path = os.path.join(cfg.DEFAULT_DATA_DIR, "window_info.json")
    with open(window_info_path, "w", encoding='utf-8') as file:
        json.dump(window_info, file)

    date_info_path = os.path.join(cfg.DEFAULT_DATA_DIR, "date_info.json")
    with open(date_info_path, "w", encoding='utf-8') as file:
        json.dump(date_info, file)

    number_of_traces_path = os.path.join(cfg.DEFAULT_DATA_DIR, "number_of_traces.json")
    with open(number_of_traces_path, "w", encoding='utf-8') as file:
        json.dump(number_of_traces, file)

    utils.generate_annotations(drift_info,
                               dir=cfg.DEFAULT_DATA_DIR,
                               log_matching=log_matching,
                               log_names=log_files.keys())

    if cfg.AUTOMATE_TFR_SCRIPT:
        utils.start_tfr_script(repo_dir=cfg.TENSORFLOW_MODELS_DIR,
                               data_dir=cfg.DEFAULT_DATA_DIR,
                               tfr_dir=cfg.TFR_RECORDS_DIR,
                               prefix=cfg.OUTPUT_PREFIX)
 

def vdd_pipeline():
    """Main function for preprocessing the event logs for the VDD approach. 
    """
    # create experiment folder structure
    cfg.DEFAULT_DATA_DIR = vdd_helper.create_experiment(cfg.DEFAULT_DATA_DIR)

    # get all paths and file names of event logs
    log_files = utils.get_event_log_paths(cfg.DEFAULT_LOG_DIR)

    drift_info = vdd_helper.extract_vdd_drift_information(cfg.DEFAULT_LOG_DIR)

    bbox_df = pd.DataFrame()

    # incrementally store number of log based on drift type - for file naming purposes
    drift_number = 1

    log_matching = {}
    date_info = {}
    first_timestamps = {}
    number_of_traces = {}

    # iterate through log files
    for name, path in tqdm(log_files.items(), desc="Preprocessing Event Logs",
                           unit="Event Log"):

        log_path = os.path.join(path, name)

        # load event log
        event_log = utils.import_event_log(path=path, name=name)

        # if the log contains incomplete traces, the log is filtered
        filtered_log = utils.filter_complete_events(event_log)

        if cfg.MINE_CONSTRAINTS:
            minerful_csv_path = vdd_helper.vdd_mine_minerful_for_declare_constraints(
                name,
                log_path,
                cfg.DEFAULT_DATA_DIR
            )
        else:
            minerful_csv_path = vdd_helper.get_minerful_constraints_path(log_name=name,
                                                                         constraints_dir=cfg.CONSTRAINTS_DIR)
            if not minerful_csv_path:
                minerful_csv_path = \
                vdd_helper.vdd_mine_minerful_for_declare_constraints(
                    name,
                    log_path,
                    cfg.DEFAULT_DATA_DIR
                )

        ts_ticks = vdd_helper.vdd_save_separately_timestamp_for_each_constraint_window(
            filtered_log)

        first_timestamps[name] = vdd_helper.get_first_timestamp_per_trace(
            filtered_log)

        number_of_traces[name] = utils.get_number_of_traces(filtered_log)

        constraints = vdd_helper.vdd_import_minerful_constraints_timeseries_data(
            minerful_csv_path)

        try:
            constraints, \
                _, \
                _, \
                _, \
                _, \
                _ = \
                vdd.do_cluster_changePoint(constraints, cp_all=cfg.CP_ALL)
        # In some edge cases the change points can not be determined
        # The error occurs only extremely rarely and is therefore skipped
        except ValueError:
            continue

        timestamps = vdd_helper.get_drift_moments_timestamps(log_name=name,
                                                             drift_info=drift_info)

        drift_types = vdd_helper.get_drift_types(log_name=name,
                                                 drift_info=drift_info)

        bboxes, log_date_info = vdd_helper.vdd_draw_drift_map_with_clusters(
            data=constraints,
            number=drift_number,
            exp_path=cfg.DEFAULT_DATA_DIR,
            ts_ticks=ts_ticks,
            timestamps=timestamps,
            drift_types=drift_types)

        bbox_df = vdd_helper.update_bboxes_for_vdd(
            bbox_df, bboxes, name)

        log_matching[name] = drift_number
        date_info[name] = log_date_info

        # increment log number
        drift_number += 1

    drift_info = vdd_helper.merge_bboxes_with_drift_info(bbox_df, drift_info)

    drift_info.to_csv(os.path.join(cfg.DEFAULT_DATA_DIR, "drift_info.csv"))

    log_matching_df = pd.DataFrame.from_dict(log_matching,
                                             orient="index",
                                             columns=["image_id"])
    log_matching_df.to_csv(os.path.join(
        cfg.DEFAULT_DATA_DIR, "log_matching.csv"))

    date_info_path = os.path.join(cfg.DEFAULT_DATA_DIR, "date_info.json")
    with open(date_info_path, "w", encoding='utf-8') as file:
        json.dump(date_info, file)

    first_timestamps_path = os.path.join(
        cfg.DEFAULT_DATA_DIR, "first_timestamps.json")
    with open(first_timestamps_path, "w", encoding='utf-8') as file:
        json.dump(first_timestamps, file)

    number_of_traces_path = os.path.join(
        cfg.DEFAULT_DATA_DIR, "number_of_traces.json")
    with open(number_of_traces_path, "w", encoding='utf-8') as file:
        json.dump(number_of_traces, file)

    vdd_helper.generate_vdd_annotations(drift_info,
                                        dir=cfg.DEFAULT_DATA_DIR,
                                        log_matching=log_matching,
                                        log_names=log_matching.keys())

    if cfg.AUTOMATE_TFR_SCRIPT:
        utils.start_tfr_script(repo_dir=cfg.TENSORFLOW_MODELS_DIR,
                                   data_dir=cfg.DEFAULT_DATA_DIR,
                                   tfr_dir=cfg.TFR_RECORDS_DIR,
                                   prefix=cfg.OUTPUT_PREFIX)


def log_to_windowed_dfg_count(event_log: EventLog, n_windows: int) \
    -> Tuple[np.ndarray,list,dict,tuple]:
    """Convert event log to directly follows frequency counts for each activity.

    Args:
        event_log (EventLog): Event log
        n_windows (int): Number of windows

    Returns:
        Tuple[np.ndarray,list,dict,tuple]: Tuple, containing the dfg frequency counts, 
        border information of windows in form of trace indices, 
        window information with first trace and timestamp and
        date information with first and last timestamp of event log
    """

    # convert event log to pandas dataframe
    event_log_df = log_converter.apply(
        event_log, variant=log_converter.Variants.TO_DATA_FRAME)
    event_log_df = dataframe_utils.convert_timestamp_columns_in_df(
        event_log_df, timest_format="ISO8601")

    if getattr(cfg, 'BASE_REPRESENTATION', 'dfg') == 'dfg_io':
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
    # hint: pandas unique does not sort the result, therefore it is faster and the
    # chronological order is maintained
    unique_traces = pd.unique(event_log_df["case:concept:name"])
    # unique_traces = utils.extract_integers_from_list(unique_traces)
    
    # get window size based on number of windows and event log size
    # event log size is equal to number of traces
    window_size = len(event_log) // n_windows

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

        # get all trace names that are in selected window, traces are sorted by
        # timestamp
        if i < n_windows:

            w_unique_traces = unique_traces[left_boundary:right_boundary]

        # at last window fill until the end of the list
        else:
            w_unique_traces = unique_traces[left_boundary:]
            right_boundary = len(unique_traces) - 1 

        # search all events for given traces
        log_window = event_log_df[event_log_df["case:concept:name"]
                                  .isin(w_unique_traces)].copy()

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
    total_freq = utils.check_dfg_graph_freq(event_log_df)
    assert total_freq == freq_count, (
        "Missing directly follow relations.\n"
        f"Number of relations in whole event log: {total_freq}.\n"
        f"Number of relations in all windows: {freq_count}"
    )
    
    return np.array(dfg_graphs), borders, window_information, date_info


def similarity_calculation(windowed_dfg: np.ndarray) -> np.ndarray:
    """Compute similarity values for each element in directly follows relations.
    Creates a quadratic matrix that can be treated as an image by normalizing 
    the similarity values.

    Args:
        windowed_dfg (np.ndarray): Array containing directly follows relations

    Returns:
        np.ndarray: Similarity matrix
    """
    # create matrix
    n = len(windowed_dfg)
    sim_matrix = np.zeros((n, n))

    # calculate similarity measure between all elements of dfg matrix
    for i, matrix_i in enumerate(windowed_dfg):
        for j, matrix_j in enumerate(windowed_dfg):
            if (i == j) or (sim_matrix[i, j] != 0):
                continue
            else:
                sim_matrix[i, j] = calc_distance_norm(matrix_i, 
                                                      matrix_j, 
                                                      cfg.DISTANCE_MEASURE)
                sim_matrix[j, i] = sim_matrix[i, j]

    # check diagonal of similarity matrix
    assert np.sum(np.diagonal(sim_matrix)
                  ) == 0, "The diagonal of the similarity matrix must be zero."

    # normalize similarity matrix
    norm_sim_matrix = 1 - sim_matrix / np.amax(sim_matrix)

    # transform matrix values to color integers
    img_matrix = np.uint8(norm_sim_matrix * 255)

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
