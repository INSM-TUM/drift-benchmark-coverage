import os
import argparse
import pandas as pd


def calc_metrics(
    csv_path: str,
    lag: float = 0.05,
    n_windows: int = 70,
    output_dir: str = None
):
    """
    Calculates overall Precision, Recall, and F1 score for CDRIFT evaluation results,
    filtering by lag and window size, and saves the summary into total_metrics.txt.
    Supports absolute paths.
    """
    csv_path = os.path.abspath(os.path.expanduser(csv_path))

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    else:
        output_dir = os.path.abspath(os.path.expanduser(output_dir))

    os.makedirs(output_dir, exist_ok=True)

    total_df = pd.read_csv(csv_path)

    # Filter by lag
    lag_df = total_df[total_df['lag'] == lag]
    if lag_df.empty:
        print(f"Warning: No rows found for lag == {lag}. Available lag values: {total_df['lag'].unique()}")
        window_df = total_df
    else:
        window_df = lag_df

    # Filter by windows if the column exists
    if 'windows' in window_df.columns:
        w_df = window_df[window_df['windows'] == n_windows]
        if not w_df.empty:
            window_df = w_df
        else:
            print(f"Warning: No rows found for windows == {n_windows}. Available window values: {window_df['windows'].unique()}")

    tp_sum = window_df['TP'].sum()
    fp_sum = window_df['FP'].sum()
    fn_tp_sum = window_df['FN_TP'].sum()

    precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    recall = tp_sum / fn_tp_sum if fn_tp_sum > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")

    metrics = {
        "tp_sum": tp_sum,
        "fp_sum": fp_sum,
        "fn_tp_sum": fn_tp_sum,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "lag": lag,
        "windows": n_windows,
    }

    total_metrics_path = os.path.join(output_dir, "total_metrics.txt")
    with open(total_metrics_path, "w") as f:
        f.write(f"TP Sum: {metrics['tp_sum']}\n")
        f.write(f"FP Sum: {metrics['fp_sum']}\n")
        f.write(f"FN_TP Sum: {metrics['fn_tp_sum']}\n")
        f.write(f"Precision: {metrics['precision']}\n")
        f.write(f"Recall: {metrics['recall']}\n")
        f.write(f"F1: {metrics['f1']}\n")
        f.write(f"Lag: {lag}\n")
        f.write(f"Windows: {n_windows}\n")

    print(f"Saved total metrics summary to {total_metrics_path}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate total metrics for CDRIFT evaluation results.")
    parser.add_argument(
        "-p", "--input_csv",
        type=str,
        required=True,
        help="Path (absolute or relative) to prediction_results_evaluated_all_windows.csv file."
    )
    parser.add_argument(
        "-l", "--lag",
        type=float,
        default=0.05,
        help="Relative lag filter (default: 0.05)."
    )
    parser.add_argument(
        "-w", "--windows",
        type=int,
        default=70,
        help="Number of windows filter (default: 70)."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default=None,
        help="Output directory to save total_metrics.txt. Defaults to the folder containing input_csv."
    )

    args = parser.parse_args()

    calc_metrics(
        csv_path=args.input_csv,
        lag=args.lag,
        n_windows=args.windows,
        output_dir=args.output_dir
    )
