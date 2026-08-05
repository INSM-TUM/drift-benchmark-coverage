import ast
from utils.cdrift_original import getTP_FP
import utils.config as cfg
import sys
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import time
from typing import Optional, Any, Dict, List, Tuple
from evaluate_scdd_4d import load_initial_drift_info_file, save_reports, log_total_time, get_recall, get_precision, get_f1_score, get_noise_info
from pathlib import Path
import json


def reorganize_drift_info(actual_drifts: List[str], actual_cp: List[Tuple[int, int]], detected_drifts: List[str], detected_cp: List[Tuple[int, int]]) -> List[Tuple[str, str, Tuple[int, int]]]:
    """
    Reorganizes drift information into a structured format for further processing.

    Parameters:
    - actual_drifts (List[str]): A list of actual drift types.
    - actual_cp (List[Tuple[int, int]]): A list of actual changepoints associated with the drift types.
    - detected_drifts (List[str]): A list of detected drift types.
    - detected_cp (List[Tuple[int, int]]): A list of detected changepoints associated with the drift types.

    Returns:
    - List[Tuple[str, str, Tuple[int, int]]]: A list of tuples containing the group type ('actual' or 'detected'),
      drift type, and associated changepoint.
    """

    reorganized_drift_info = []

    # Create tuples for actual changepoints
    for drift_type, changepoint in zip(actual_drifts, actual_cp):
        reorganized_drift_info.append(("actual", drift_type, changepoint))

    # Create tuples for detected changepoints
    for drift_type, changepoint in zip(detected_drifts, detected_cp):
        reorganized_drift_info.append(("detected", drift_type, changepoint))

    return reorganized_drift_info



def process_change_points_level(row: pd.Series,
                                drift_info_df_noise: pd.DataFrame,
                                LAG: int,
                                log_sizes) -> Dict[str, Any]:
    """
    Processes a row of change point data and evaluates detected change points against actual change points.

    Parameters:
    - row (pd.Series): A row from the DataFrame containing log evaluation data.
    - drift_info_df_noise (pd.DataFrame): DataFrame containing noise information for drift analysis.
    - LAG (int): The lag value used for evaluating change points.

    Returns:
    - Dict[str, Any]: A dictionary containing the evaluation results, including log name, noise level,
                      actual and detected change points, and true/false positives.
    """

    log_size = log_sizes.loc[log_sizes["log_name"] == row["Log"] + ".xes", ["n_traces"]].values[0].tolist()[0]

    lag_acc = int(log_size * LAG)  # Calculate lag accuracy


    # Get actual and detected change point info
    actual_cp: List[int] = ast.literal_eval(row["Actual Changepoints for Log"])
    detected_cp: List[int] = ast.literal_eval(row["Detected Changepoints"])

    # Calculate true positives and false positives
    TP, FP = getTP_FP(detected_cp, actual_cp, lag_acc)
    FN_TP = len(actual_cp)  # Number of actual change points

    # Prepare the evaluation row as a dictionary
    evaluation_row: Dict[str, Any] = {
        'Algorithm': row['Algorithm'],
        'log_name': row['Log'],
        'noise_level': get_noise_info(row['Log'] , drift_info_df_noise),
        'drift_type': 'NA',
        'actual_cp': actual_cp,
        'detected_cp': detected_cp,
        'log_size': log_size,
        'lag': LAG,
        'lag_indices': lag_acc,
        'TP': TP,
        'FP': FP,
        'FN_TP': FN_TP
    }

    return evaluation_row


def evaluation_change_point_level_in_parallel(full_path: str, LAG: int,
                                              num_cores: Optional[int] = None) -> pd.DataFrame:
    """
    Evaluates change point levels in parallel processing using multiple CPU cores.

    Parameters:
    - full_path (str): The file path to the CSV file containing evaluation information.
    - LAG (int): The lag value used in processing change points.
    - num_cores (Optional[int]): The number of CPU cores to use for parallel processing. Defaults to None, which uses all available cores.

    Returns:
    - pd.DataFrame: A DataFrame containing the evaluation report of change points.
    """

    # Load CSV file with evaluation info
    df = pd.read_csv(full_path)

    # Load initial drift info file
    drift_info_df = load_initial_drift_info_file(cfg.DRIFT_INFO_INITIAL)
    drift_info_df_noise = drift_info_df.loc[drift_info_df["drift_sub_attribute"] == "noisy_trace_prob"]

    # Prepare the evaluation report DataFrame
    evaluation_report = pd.DataFrame()

    # Total number of tasks to process
    total_tasks = len(df)


    with open(JSON_PATH) as f:
        data = json.load(f)
    log_sizes = pd.DataFrame(list(data.items()), columns=['log_name', 'n_traces'])

    # Use ProcessPoolExecutor to parallelize the processing of logs using multiple CPU cores
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit tasks for parallel execution and store results as futures
        futures = [executor.submit(process_change_points_level, row, drift_info_df_noise, LAG, log_sizes) for _, row in
                   df.iterrows()]

        # Initialize a counter for completed tasks
        completed_tasks = 0

        # As futures are completed, collect results and concatenate into the DataFrame
        for future in futures:
            evaluation_row = future.result()
            evaluation_report = pd.concat([evaluation_report, pd.DataFrame.from_records([evaluation_row])],
                                          ignore_index=True)

            # Increment the completed tasks counter
            completed_tasks += 1

            # Calculate and print progress, overwriting the same line
            progress_percentage = (completed_tasks / total_tasks) * 100
            sys.stdout.write(f"\rProgress change point level: {progress_percentage:.2f}% completed.")
            sys.stdout.flush()  # Ensure the output is written immediately

    print()  # Print a new line after completion
    return evaluation_report


def main_change_points(base_path_main: str, num_cores: int):
    """
    Function to process relative lags and generate the evaluation report for change points.

    Parameters:
    - base_path_main (str): The base directory where the evaluation files are stored.
    - num_cores (int): Number of cores to use for parallel processing.
    """
    start_time = time.time()  # Start tracking time
    combined_evaluation_report_cp = []  # Initialize list for change point reports

    for lag in cfg.RELATIVE_LAG:
        print(f"Lag in progress (change points): {lag}")

        # Get evaluation report for the current lag
        lag_start_time = time.time()

        full_path = os.path.join(base_path_main, file_name)

        # Generate evaluation reports using parallel processing
        evaluation_report_cp = evaluation_change_point_level_in_parallel(full_path, LAG=lag, num_cores=num_cores)

        # Log time taken for this lag
        elapsed_time = time.time() - lag_start_time

        # Add lag as a new column
        evaluation_report_cp['lag'] = lag
        combined_evaluation_report_cp.append(evaluation_report_cp)  # Append current lag results
        save_reports(evaluation_report_cp, base_path_main, f'evaluation_cdrift_cp_{lag}.csv')

        print(f"Elapsed time for lag {lag}: {elapsed_time:.2f} seconds\n")

    # Combine all change point reports into a single DataFrame
    final_evaluation_report_cp = pd.concat(combined_evaluation_report_cp, ignore_index=True)

    # Save the report
    save_reports(final_evaluation_report_cp, base_path_main, f'evaluation_cdrift_cp.csv')

    # Aggregate and summarize change point reports
    grouping = [["noise_level", "Algorithm"], ["lag", "Algorithm"]]
    summarize_results(final_evaluation_report_cp, base_path_main, grouping, 'evaluation_cdrift_cp_agg.csv')

    # Log total time taken
    log_total_time(start_time)


def get_total_evaluation_results(evaluation_report, grouping):
    aggregations = {
        'TP': 'sum',
        'FP': 'sum',
        'FN_TP': 'sum'}
    # grouping = ['noise_level', 'complexity', 'method', 'window_size', 'lag', 'eval_focus']
    #grouping = ['noise_level', 'complexity', 'n_drifts', "n_change_points", 'drift_type']
    #pdb.set_trace()

    # Convert Series objects to strings
    evaluation_report['noise_level'] = evaluation_report['noise_level'].astype(str)

    evaluation_report_agg = evaluation_report.groupby(grouping).agg(aggregations)

    evaluation_report_agg = evaluation_report_agg.assign(Precision=lambda x: get_precision(x['TP'], x['FP']))
    evaluation_report_agg = evaluation_report_agg.assign(Recall=lambda x: get_recall(x['TP'], x['FN_TP']))
    evaluation_report_agg = evaluation_report_agg.assign(F1_score=lambda x: get_f1_score(x['Precision'], x['Recall']))

    return evaluation_report_agg



def summarize_results(results_df, base_path_main, grouping, file_name):

    for group in grouping:
        result = get_total_evaluation_results(results_df, group)

        # Pivot the DataFrame
        df_pivoted = result[['Precision', 'Recall', 'F1_score']].unstack(level=group[0])

        if len(group) == 2:
            # Sort the columns
            df_pivoted.columns = df_pivoted.columns.swaplevel(0, 1)
            df_pivoted = df_pivoted.sort_index(level=0, axis=1)

            new_order = []
            for lag in sorted(set(df_pivoted.columns.get_level_values(0))):  # Get unique lags and sort them
                new_order.append((lag, 'Precision'))
                new_order.append((lag, 'Recall'))
                new_order.append((lag, 'F1_score'))

            # Create the new MultiIndex
            new_index = pd.MultiIndex.from_tuples(new_order, names=[group[0], None])
            df_pivoted = df_pivoted[new_index]


            if group[1] == "drift_type":
                index = ['no_drift', 'sudden', 'gradual', 'incremental', 'recurring']
                df_pivoted = pd.DataFrame(df_pivoted, index=index)

            elif group[1] == "noise_level":
                index = ['0.0', '0.3', '0.6']
                df_pivoted = pd.DataFrame(df_pivoted, index=index)
            else:
                pass
        else:
            # Reset the index to convert the Series to DataFrame
            df_reset = df_pivoted.reset_index()

            # Rename columns for clarity
            df_reset.columns = ['Metric', 'lag', 'Value']

            # Set the new index with lag and Metric
            df_swapped = df_reset.set_index(['lag', 'Metric'])

            # Sort the index levels
            df_swapped = df_swapped.sort_index(level=['lag', 'Metric'])

            new_order = []
            for lag in set(df_pivoted.index.get_level_values(1)):  # Get unique lags and sort them
                new_order.append((lag, 'Precision'))
                new_order.append((lag, 'Recall'))
                new_order.append((lag, 'F1_score'))

            sorted_tuples = sorted(new_order, key=lambda x: (x[0]))

            # Create a MultiIndex from the new_index list
            new_multi_index = pd.MultiIndex.from_tuples(sorted_tuples)

            df_pivoted = df_swapped.reindex(new_multi_index).T


        focus = "_".join(group)
        file_name_new = file_name.split('.')[0] + '_' + focus+ ".csv"
        report_path = os.path.join(base_path_main, file_name_new)
        df_pivoted.to_csv(report_path, index=True)
        print(f"Aggregated reports saved {focus}: {report_path}")
        print(df_pivoted)
        # Round values to 2 decimal places
        df_rounded = df_pivoted.round(2)

        # Convert to LaTeX format
        latex_output = df_rounded.to_latex(index=True, header=True, float_format="%.2f")

        print(latex_output)
        print()



if __name__ == "__main__":
    num_cores = 150
    file_name = "algorithm_results_scdd_test_v1.csv"

    #JSON_PATH = Path("C:\\Users\\alkraus\\ResearchGit\\cv4cdd_main\\data\\input_4d\\test\\winsim\\experiment_20240919-173212\\number_of_traces.json")
    #base_path_main = os.path.join("C:\\Users\\alkraus\\ResearchGit\\cv4cdd_cdrift\\Evaluation_Results_SCDD_4d\\Reproducibility_Intermediate_Results_test_v1")

    JSON_PATH = Path("/work/alexkrau/projects/scdd/data/input_4d/test/winsim/experiment_20240919-173212/number_of_traces.json")
    base_path_main = Path("/work/alexkrau/projects/cdrift-evaluation/Evaluation_Results_SCDD_4d/Reproducibility_Intermediate_Results_test_v1")

    main_change_points(base_path_main, num_cores)
