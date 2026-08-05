import pdb
import pandas as pd
import ast
from utils.cdrift_original import getTP_FP
import numpy as np
import tqdm
import utils.config as cfg
import os
import json
from typing import List, Tuple, Union
import os
import pandas as pd
from multiprocessing import Pool, cpu_count


def _weighted_f1_score(group):
    fn_tp_weighted_f1 = (group['F1_score'] * group['FN_TP']).sum() / group['FN_TP'].sum()
    return pd.Series({'Weighted_F1': fn_tp_weighted_f1})



def _get_weighted_F1_accuracy(report):
    # Group by 'noise_level', 'lag', and any other columns that are constant for each group
    grouped = report.groupby(['noise_level', "drift_type"])

    # Apply the custom function to calculate the weighted F1 score for each group
    weighted_F1_per_set = grouped.apply(_weighted_f1_score).reset_index()

    return weighted_F1_per_set


def get_total_evaluation_results(evaluation_report):
    aggregations = {
        'TP': 'sum',
        'FP': 'sum',
        'FN_TP': 'sum'}
    # grouping = ['noise_level', 'complexity', 'method', 'window_size', 'lag', 'eval_focus']
    #grouping = ['noise_level', 'complexity', 'n_drifts', "n_change_points", 'drift_type']
    #pdb.set_trace()

    grouping = ["noise_level", "drift_type"]
    # Convert Series objects to strings
    evaluation_report['noise_level'] = evaluation_report['noise_level'].astype(str)
    evaluation_report['drift_type'] = evaluation_report['drift_type'].astype(str)

    evaluation_report_agg = evaluation_report.groupby(grouping).agg(aggregations)

    evaluation_report_agg = evaluation_report_agg.assign(Precision=lambda x: get_precision(x['TP'], x['FP']))
    evaluation_report_agg = evaluation_report_agg.assign(Recall=lambda x: get_recall(x['TP'], x['FN_TP']))
    evaluation_report_agg = evaluation_report_agg.assign(F1_score=lambda x: get_f1_score(x['Precision'], x['Recall']))

    return evaluation_report_agg


def transform_tuples(original_list):
    """
    Transforms a list of tuples containing start and end values along with a drift type
    into a new list of tuples where each start value is paired with 'drift_type_start'
    and each end value is paired with 'drift_type_end'.

    Args:
    original_list (list): A list of tuples in the format [((start1, end1), drift_type1), ((start2, end2), drift_type2), ...]

    Returns:
    list: A new list of tuples in the format [(start1, 'drift_type1_start'), (end1, 'drift_type1_end'), ...]
    """
    new_list = []

    # Iterate over the original list of tuples
    for item in original_list:
        # Extract start and end values from the tuple
        start, end = item[0]
        drift_type = item[1]
        if item[1] == "gradual":
            # Create new tuples with gradual start and end infos
            new_list.append((start, drift_type + '_start'))
            new_list.append((end, drift_type + '_end'))
        else:
            # Create new tuples with sudden infos
            new_list.append((start, drift_type))

    return new_list

def add_no_drift_labels(drift_info: list):

    if len(drift_info) == 0:
        drift_info.append([0, "no_drift"])

    return drift_info


def get_precision(TP, FP):
    precision = np.where(TP + FP > 0, np.divide(TP, TP + FP), 0)
    return precision


def get_recall(TP, FN_TP):
    recall = np.where(FN_TP > 0, np.divide(TP, FN_TP), 0)
    return recall


def get_f1_score(precision, recall):

    f1_score = np.where(precision + recall > 0,  (2 * precision * recall) / (precision + recall), 0)

    return f1_score


def count_number_of_change_points(drift_patter):
    count_sudden = 0
    count_gradual = 0

    for item in drift_patter:
        if item == "sudden":
            count_sudden += 1
        else:
            count_gradual += 1

    return (count_sudden, count_gradual)

def load_initial_drift_info_file(data_dir: str) -> pd.DataFrame:
    """Loads drift info from file.

    Args:
        data_dir (str): Directory where drift info is stored

    Returns:
        pd.DataFrame: Drift info
    """
    drift_info_path = os.path.join(data_dir, "drift_info.csv")
    assert os.path.isfile(drift_info_path), "No drift info file found"
    return pd.read_csv(drift_info_path, sep=";")


def get_noise_info(log_name: str, noise_info: pd.DataFrame) -> pd.DataFrame:
    """Get noise info for event log.

    Args:
        log_name (str): Name of event log
        noise_info (pd.DataFrame): Drift info (filtered)

    Returns:
        pd.DataFrame: DataFrame, containing drift info for event log
    """
    #pdb.set_trace()
    value = str(0.0)
    if log_name in noise_info["log_name"].tolist():
        value = noise_info.loc[noise_info["log_name"] == log_name]["value"].values[0]

    return value


def get_traces_per_log(data_dir: str) -> Union[list, dict]:
    """Load traces per log from file.


    Args:
        data_dir (str): Directory where traces per log info is stored

    Returns:
        Union[list, dict]: Number of traces per log
    """
    number_of_traces_path = os.path.join(data_dir, "number_of_traces.json")
    assert os.path.isfile(
        number_of_traces_path), "No number of traces file found"
    file = open(number_of_traces_path)
    return json.load(file)



def main_cp_only_external(full_path, LAG):

    # Load csv file with evaluation info
    df = pd.read_csv(full_path)

    # Add actual drift info depending on the dataset from path:
    if os.path.basename(os.path.split(full_path)[0]) == "ostovar":
        df['Actual Changepoints'] = str([(999, 999), (1999, 1999)])
        log_size = 2999

    evaluation_report = pd.DataFrame()
    for index, row in df.iterrows():
        log_name = row.iloc[0]
        print(f"WIP: {index}: {log_name}")

        # get actual and detected change point infos
        actual_cp = ast.literal_eval(row["Actual Changepoints"])
        detected_cp = [(int(x), int(y)) for x, y in ast.literal_eval(row["Detected Changepoints"])]

        actual_info_rel = [item[0] for item in actual_cp]
        detected_info_rel = [item[0] for item in detected_cp if item[0] == item[1]]

        lag_acc = int(log_size * LAG)

        TP, FP = getTP_FP(detected_info_rel, actual_info_rel, lag_acc)
        FN_TP = len(actual_info_rel)
        evaluation_row = {'log_name': log_name,
                          'complexity': "na",
                          'actual_cp': actual_info_rel,
                          'detected_cp': detected_info_rel,
                          'log_size': log_size,
                          'lag_indices': lag_acc,
                          'TP': TP,
                          'FP': FP,
                          'FN_TP': FN_TP,
                          'lag': LAG,
                           }
        evaluation_report = pd.concat([evaluation_report, pd.DataFrame.from_records([evaluation_row])],
                                      ignore_index=True)

    return evaluation_report



def main_cp_only(full_path, LAG):


    # Load csv file with evaluation info
    df = pd.read_csv(full_path)

    # Load traces per log
    traces_per_log = get_traces_per_log(cfg.TEST_IMAGE_DATA_DIR)

    # Load initial drift info file
    drift_info_df = load_initial_drift_info_file(cfg.DRIFT_INFO_INITIAL)
    drift_info_df_noise = drift_info_df.loc[drift_info_df["drift_sub_attribute"] == "noisy_trace_prob"]

    # Create evaluation dictionary
    drift_types = ['no_drift', "sudden", "gradual_start", "gradual_end"]

    evaluation_report = pd.DataFrame()
    for index, row in df.iterrows():
        log_name = row.iloc[0]
        print(f"WIP: {index}: {log_name}")
        # select noise level used for the given log
        noise_level = get_noise_info(log_name, drift_info_df_noise)

        # get actual and detected change point infos
        actual_cp = ast.literal_eval(row["Actual Changepoints"])
        drift_pattern = ast.literal_eval(row["Actual Drift Types"])
        n_drifts = len(drift_pattern)
        n_sudden, n_gradual = count_number_of_change_points(drift_pattern)
        actual_drift = tuple(drift_pattern)
        actual_info = list(zip(actual_cp, actual_drift))
        actual_info = transform_tuples(actual_info)
        n_change_points = len(actual_info)
        actual_info = add_no_drift_labels(actual_info)

        detected_cp = ast.literal_eval(row["Detected Changepoints"])
        detected_drift = ast.literal_eval(row["Predicted Drift Types"])
        detected_info = list(zip(detected_cp, detected_drift))
        detected_info = transform_tuples(detected_info)
        detected_info = add_no_drift_labels(detected_info)

        actual_info_rel = [item[0] for item in actual_info]
        detected_info_rel = [item[0] for item in detected_info]
        log_size = traces_per_log[log_name]
        lag_acc = int(log_size * LAG)
        TP, FP = getTP_FP(detected_info_rel, actual_info_rel, lag_acc)
        FN_TP = len(actual_info_rel)
        evaluation_row = {'log_name': log_name,
                          # TODO: complexity info:
                          'noise_level': noise_level,
                          'complexity': "na",
                          'drift_pattern': str(drift_pattern),
                          'n_drifts': n_drifts,
                          'n_change_points': n_change_points,
                          'n_sudden': n_sudden,
                          'n_gradual': n_gradual,
                          'drift_type': 'NA',
                          'actual_cp': actual_info_rel,
                          'detected_cp': detected_info_rel,
                          'log_size': log_size,
                          'lag': LAG,
                          'lag_indices': lag_acc,
                          'TP': TP,
                          'FP': FP,
                          'FN_TP': FN_TP,
                          'lag': LAG,
                           }
        #print(evaluation_row)
        evaluation_report = pd.concat([evaluation_report, pd.DataFrame.from_records([evaluation_row])],
                                      ignore_index=True)

    evaluation_report.to_csv(os.path.join(os.path.dirname(full_path), f'{os.path.basename(full_path)[:-4]}_evaluated.csv'), index=True)

    return evaluation_report



def main(full_path, LAG):


    # Load csv file with evaluation info
    df = pd.read_csv(full_path)

    # Load traces per log
    traces_per_log = get_traces_per_log(cfg.TEST_IMAGE_DATA_DIR)

    # Load initial drift info file
    drift_info_df = load_initial_drift_info_file(cfg.DRIFT_INFO_INITIAL)
    drift_info_df_noise = drift_info_df.loc[drift_info_df["drift_sub_attribute"] == "noisy_trace_prob"]

    # Create evaluation dictionary
    drift_types = ['no_drift', "sudden", "gradual_start", "gradual_end"]

    evaluation_report = pd.DataFrame()
    for index, row in df.iterrows():
        log_name = row.iloc[0]
        print(f"WIP: {index}: {log_name}")
        # select noise level used for the given log
        noise_level = get_noise_info(log_name, drift_info_df_noise)

        # get actual and detected change point infos
        actual_cp = ast.literal_eval(row["Actual Changepoints"])
        drift_pattern = ast.literal_eval(row["Actual Drift Types"])
        n_drifts = len(drift_pattern)
        n_sudden, n_gradual = count_number_of_change_points(drift_pattern)
        actual_drift = tuple(drift_pattern)
        actual_info = list(zip(actual_cp, actual_drift))
        actual_info = transform_tuples(actual_info)
        n_change_points = len(actual_info)
        actual_info = add_no_drift_labels(actual_info)

        detected_cp = ast.literal_eval(row["Detected Changepoints"])
        detected_drift = ast.literal_eval(row["Predicted Drift Types"])
        detected_info = list(zip(detected_cp, detected_drift))
        detected_info = transform_tuples(detected_info)
        detected_info = add_no_drift_labels(detected_info)

        for drift_type in drift_types:
            actual_info_rel = [item[0] for item in actual_info if item[1] == drift_type]
            detected_info_rel = [item[0] for item in detected_info if item[1] == drift_type]

            log_size = traces_per_log[log_name]
            lag_acc = int(log_size * LAG)

            TP, FP = getTP_FP(detected_info_rel, actual_info_rel, lag_acc)
            FN_TP = len(actual_info_rel)
            evaluation_row = {'log_name': log_name,
                              'noise_level': noise_level,
                              'complexity': "na",
                              'drift_pattern': str(drift_pattern),
                              'n_drifts': n_drifts,
                              'n_change_points': n_change_points,
                              'n_sudden': n_sudden,
                              'n_gradual': n_gradual,
                              'drift_type': drift_type,
                              'actual_cp': actual_info_rel,
                              'detected_cp': detected_info_rel,
                              'log_size': log_size,
                              'lag': LAG,
                              'lag_indices': lag_acc,
                              'TP': TP,
                              'FP': FP,
                              'FN_TP': FN_TP
                               }
            evaluation_report = pd.concat([evaluation_report, pd.DataFrame.from_records([evaluation_row])],
                                          ignore_index=True)

    # Calculate aggregated measures
    #evaluation_report_agg = get_total_evaluation_results(evaluation_report)
    #evaluation_report_agg_weighted = _get_weighted_F1_accuracy(evaluation_report_agg)
    #weighted_F1_per_set_avg = round(np.mean(evaluation_report_agg_weighted.Weighted_F1.values) * 100, 2)

    # Save results to a table
    evaluation_report.to_csv(os.path.join(os.path.dirname(full_path), f'{os.path.basename(full_path)[:-4]}_evaluated.csv'), index=True)
    #evaluation_report_agg.to_csv(os.path.join(base_path, f'{file_name[:-4]}_evaluated_agg.csv'), index=True)
    #evaluation_report_agg_weighted.to_csv(os.path.join(base_path, f'{file_name[:-4]}_evaluated_agg_weighted_{weighted_F1_per_set_avg}.csv'), index=True)

    return evaluation_report


def get_confusion_matrix(evaluation_report):
    actual_classes = []
    detected_classes = []
    skipped_rows = 0
    ignored_rows = 0
    for index, row in evaluation_report.iterrows():
        cp_actual = ast.literal_eval(row['actual_cp'])
        cp_detected = ast.literal_eval(row['detected_cp'])
        drift_type = row['drift_type']
        if cp_actual == cp_detected and len(cp_actual) == 0:
            # this row is not relevant at all
            skipped_rows += 1
            continue
        elif cp_actual == cp_detected and 0 in cp_actual and 0 in cp_detected:
            # this is a correctly detected "no_drift" case
            actual_classes.extend(['no_drift'])
            detected_classes.extend(['no_drift'])
        elif len(cp_actual) == 1 and len(cp_detected) == 1:
            # This is a correctly detected single instance of a drift type
            actual_classes.extend([drift_type]*1)
            detected_classes.extend([drift_type]*1)
        elif len(cp_actual) == 2 and len(cp_detected) == 2:
            # This is a correctly detected single instance of a drift type
            actual_classes.extend([drift_type]*2)
            detected_classes.extend([drift_type]*2)
        elif len(cp_actual) == 3 and len(cp_detected) == 3:
            # This is a correctly detected single instance of a drift type
            actual_classes.extend([drift_type]*3)
            detected_classes.extend([drift_type]*3)
        elif len(cp_actual) == 1 and len(cp_detected) == 0:
            # This is a not detected single instance of a drift type
            actual_classes.extend([drift_type]*1)
            detected_classes.extend(['no_drift']*1)
        elif len(cp_actual) == 0 and len(cp_detected) == 1:
            # This is a false positive single instance of a drift type
            actual_classes.extend(['no_drift']*1)
            detected_classes.extend([drift_type]*1)
        elif len(cp_actual) == 2 and len(cp_detected) == 0:
            # This is a not detected single instance of a drift type
            actual_classes.extend([drift_type]*2)
            detected_classes.extend(['no_drift']*2)
        elif len(cp_actual) == 0 and len(cp_detected) == 2:
            # This is a false positive single instance of a drift type
            actual_classes.extend(['no_drift']*2)
            detected_classes.extend([drift_type]*2)
        elif len(cp_actual) == 3 and len(cp_detected) == 0:
            # This is a not detected single instance of a drift type
            actual_classes.extend([drift_type]*3)
            detected_classes.extend(['no_drift']*3)
        elif len(cp_actual) == 0 and len(cp_detected) == 3:
            # This is a false positive single instance of a drift type
            actual_classes.extend(['no_drift']*3)
            detected_classes.extend([drift_type]*3)
        elif len(cp_actual) == 3 and len(cp_detected) == 1:
            # This is a not detected single instance of a drift type
            actual_classes.extend([drift_type]*3)
            detected_classes.extend([drift_type] + ['no_drift']*2)
        elif len(cp_actual) == 1 and len(cp_detected) == 3:
            # This is a false positive single instance of a drift type
            actual_classes.extend([drift_type] + ['no_drift']*2)
            detected_classes.extend([drift_type]*3)
        elif len(cp_actual) == 2 and len(cp_detected) == 1:
            # This is a not detected single instance of a drift type
            actual_classes.extend([drift_type]*2)
            detected_classes.extend([drift_type] + ['no_drift']*1)
        elif len(cp_actual) == 1 and len(cp_detected) == 2:
            # This is a false positive single instance of a drift type
            actual_classes.extend([drift_type] + ['no_drift']*1)
            detected_classes.extend([drift_type]*2)
        elif len(cp_actual) == 3 and len(cp_detected) == 2:
            # This is a not detected single instance of a drift type
            actual_classes.extend([drift_type]*3)
            detected_classes.extend([drift_type]*2 + ['no_drift']*1)
        elif len(cp_actual) == 2 and len(cp_detected) == 3:
            # This is a false positive single instance of a drift type
            actual_classes.extend([drift_type]*2 + ['no_drift']*1)
            detected_classes.extend([drift_type]*3)

        else:
            ignored_rows += 1

    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt



    labels = ['no_drift', 'sudden', 'gradual_start', 'gradual_end']
    confusion_matrix = confusion_matrix(actual_classes, detected_classes, labels=labels)

    report = classification_report(actual_classes, detected_classes, output_dict=True,  labels=labels)

    # #confusion_matrixs = multilabel_confusion_matrix(actual_classes, detected_classes, labels=labels)
    #     disp = ConfusionMatrixDisplay(confusion_matrix, display_labels=labels)
    #     disp.plot(include_values=True, cmap="viridis", ax=None, xticks_rotation="vertical")
    #     plt.show()
    #
    #
    #
    # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
    # disp.plot()
    # plt.title("My confusion matrix")
    # plt.show()


    return None





def process_lag(args):
    prodrift_option, LAG = args
    path = os.path.join(base_path_main, prodrift_option)
    if os.path.basename(path)== "ostovar":
        file_name = "prediction_results.csv"
    else:
        file_name = f"evaluation_results_general_{LAG}_lag.csv"
    full_path = os.path.join(path, file_name)
    report = main(full_path, LAG)
#    report = main_cp_only(full_path, LAG)
#    report = main_cp_only_external(full_path, LAG)
#    report['technique'] = prodrift_option.split("_")[0]
#    report['window'] = prodrift_option.split("_")[1]
#    report['technique'] = 'scdd'
#    report['window'] = 'na'
#    report['lag'] = LAG


    return report


if __name__ == "__main__":

#    base_path_main = "/work/alexkrau/projects/scdd/data/output/20240226-161106_winsim_sgd/evaluation/general/" #m1
#    base_path_main = "/work/alexkrau/projects/scdd/data/output/20240228-174145_winsim_sgd/evaluation/general/" #m2
#    base_path_main = "/work/alexkrau/projects/scdd/data/output/20240228-231113_winsim_sgd/evaluation/general/" #m3
#    base_path_main = "/work/alexkrau/projects/scdd/data/output/20240301-235147_winsim_sgd/evaluation/general/" #m4
    base_path_main = "/work/alexkrau/projects/scdd/data/output/20240302-004101_winsim_sgd/evaluation/general/" #m5

    prodrift_options = ["threshold_0.5"]

    all_args = [(prodrift_option, LAG) for prodrift_option in prodrift_options for LAG in cfg.RELATIVE_LAG]

    num_cores = 10


    with Pool(processes=num_cores) as pool:
        reports = pool.map(process_lag, all_args)

    report_all = pd.concat(reports, ignore_index=True)
    report_all.to_csv(os.path.join(base_path_main, 'evaluation_results.csv'), index=True)
