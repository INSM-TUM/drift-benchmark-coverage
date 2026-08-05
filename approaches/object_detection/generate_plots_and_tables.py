import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))


def generate_benchmark_plots():
    base_dir = os.path.join(repo_root, "results", "aggregated")
    fig_dir = os.path.join(repo_root, "our_paper", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=1.1)

    color_map = {
        "DFG": "#1f77b4",       # Blue
        "DFG-IO": "#ff7f0e",    # Orange
        "ARM (Ours)": "#2ca02c" # Green
    }
    label_map = {
        "dfg": "DFG",
        "dfg_io": "DFG-IO",
        "arm": "ARM (Ours)"
    }

    # 1. Plot 1: CDLG Overall F1 across Latencies
    cdlg_overall_path = os.path.join(base_dir, "cdlg_overall.csv")
    if os.path.exists(cdlg_overall_path):
        df = pd.read_csv(cdlg_overall_path)
        df["Method"] = df["method"].map(label_map)
        df["Latency (%)"] = (df["lag"] * 100).apply(lambda x: f"{x:g}%")

        plt.figure(figsize=(8, 5))
        ax = sns.barplot(
            data=df,
            x="Latency (%)",
            y="f1",
            hue="Method",
            palette=color_map
        )
        plt.title("CDLG Change Point Detection F1-Score by Latency Threshold", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Latency Threshold (γ)", fontsize=12, fontweight="bold")
        plt.ylabel("F1-Score", fontsize=12, fontweight="bold")
        plt.ylim(0.0, 1.22)
        plt.legend(title="Representation", frameon=True, loc="upper center", ncol=3)

        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f"{height:.3f}",
                            (p.get_x() + p.get_width() / 2., height / 2.0),
                            ha='center', va='center', fontsize=10, color='white', fontweight='bold',
                            rotation=90)

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "cdlg_overall_f1.pdf"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(fig_dir, "cdlg_overall_f1.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated {os.path.join(fig_dir, 'cdlg_overall_f1.pdf')}")

    # 2. Plot 2: CDLG Noise Robustness (at 5% Latency)
    cdlg_noise_path = os.path.join(base_dir, "cdlg_noise.csv")
    if os.path.exists(cdlg_noise_path):
        df = pd.read_csv(cdlg_noise_path)
        df["Method"] = df["method"].map(label_map)
        df["Noise (%)"] = (df["noise"] * 100).astype(int).astype(str) + "%"

        plt.figure(figsize=(8, 5))
        ax = sns.lineplot(
            data=df,
            x="Noise (%)",
            y="f1",
            hue="Method",
            marker="o",
            markersize=8,
            linewidth=2.5,
            palette=color_map
        )
        plt.title("Impact of Synthetic Noise on Detection F1-Score (CDLG, γ = 5%)", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Noise Level (% of traces affected)", fontsize=12, fontweight="bold")
        plt.ylabel("F1-Score", fontsize=12, fontweight="bold")
        plt.ylim(0.5, 1.08)
        plt.legend(title="Representation", frameon=True, loc="upper center", ncol=3)

        for _, row in df.iterrows():
            m = row["Method"]
            n = row["Noise (%)"]
            val = row["f1"]
            if n == "0%":
                offset = 0.012 if m == "DFG-IO" else (-0.018 if m == "DFG" else -0.038)
                va = 'bottom' if m == "DFG-IO" else 'top'
            else:
                offset = 0.012 if "DFG" in m else -0.020
                va = 'bottom' if "DFG" in m else 'top'
            ax.annotate(f"{val:.3f}",
                        (n, val + offset),
                        ha='center', va=va, fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "cdlg_noise_robustness.pdf"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(fig_dir, "cdlg_noise_robustness.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated {os.path.join(fig_dir, 'cdlg_noise_robustness.pdf')}")

    # 3. Plot 3: CDLG Drift Types Breakdown (at 5% Latency)
    cdlg_dt_path = os.path.join(base_dir, "cdlg_drift_types.csv")
    if os.path.exists(cdlg_dt_path):
        df = pd.read_csv(cdlg_dt_path)
        df = df[df["drift_type"] != "no_drifts"].copy()
        df["Method"] = df["method"].map(label_map)
        df["Drift Type"] = df["drift_type"].str.replace("_", " ").str.title()

        plt.figure(figsize=(9, 5))
        ax = sns.barplot(
            data=df,
            x="Drift Type",
            y="f1",
            hue="Method",
            palette=color_map
        )
        plt.title("Detection F1-Score by Concept Drift Type (CDLG, γ = 5%)", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Drift Category", fontsize=12, fontweight="bold")
        plt.ylabel("F1-Score", fontsize=12, fontweight="bold")
        plt.ylim(0.0, 1.25)
        plt.legend(title="Representation", frameon=True, loc="upper center", ncol=3)

        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f"{height:.3f}",
                            (p.get_x() + p.get_width() / 2., height / 2.0),
                            ha='center', va='center', fontsize=9, color='white', fontweight='bold',
                            rotation=90)

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "cdlg_drift_types.pdf"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(fig_dir, "cdlg_drift_types.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated {os.path.join(fig_dir, 'cdlg_drift_types.pdf')}")

    # 4. Plot 4: CDRIFT Window Size Sensitivity Analysis (at 5% Latency)
    cdrift_win_path = os.path.join(base_dir, "cdrift_windows.csv")
    if os.path.exists(cdrift_win_path):
        df = pd.read_csv(cdrift_win_path)
        df["Method"] = df["method"].map(label_map)

        plt.figure(figsize=(9, 5))
        ax = sns.lineplot(
            data=df,
            x="windows",
            y="f1",
            hue="Method",
            style="Method",
            markers=True,
            dashes=False,
            markersize=7,
            linewidth=2.2,
            palette=color_map
        )
        plt.title("CDRIFT Window Sensitivity: F1-Score vs Window Count (γ = 5%)", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Number of Windows (N)", fontsize=12, fontweight="bold")
        plt.ylabel("F1-Score", fontsize=12, fontweight="bold")
        plt.ylim(0.0, 1.25)
        plt.xticks(range(60, 210, 10))
        plt.legend(title="Representation", frameon=True, loc="upper center", ncol=3)

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "cdrift_windows_sensitivity.pdf"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(fig_dir, "cdrift_windows_sensitivity.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated {os.path.join(fig_dir, 'cdrift_windows_sensitivity.pdf')}")


def generate_custom_eval_plots():
    comp_csv_path = os.path.join(repo_root, "outputs", "human_vs_cv_comparison.csv")
    fig_dir = os.path.join(repo_root, "our_paper", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    if not os.path.exists(comp_csv_path):
        print(f"File not found: {comp_csv_path}")
        return

    df = pd.read_csv(comp_csv_path)

    dataset_name_map = {
        "between_activities": "Between Activities",
        "big": "Big Processes",
        "gradual": "Gradual",
        "isolated": "Isolated",
        "loops": "Loops",
        "related_conditions_traces": "Related Conditions"
    }
    df["Dataset_Clean"] = df["Dataset"].map(dataset_name_map)

    rep_map = {
        "DFG": "DFG",
        "DFG_IO": "DFG-IO",
        "ARM": "ARM (Ours)"
    }
    df["Representation_Clean"] = df["Representation"].map(rep_map)

    sns.set_theme(style="whitegrid", font_scale=1.1)

    color_map = {
        "DFG": "#1f77b4",       # Blue
        "DFG-IO": "#ff7f0e",    # Orange
        "ARM (Ours)": "#2ca02c" # Green
    }

    # 1. Figure 1: Human Visual Inspection vs CV Model F1-Score per Dataset (ARM Representation)
    plt.figure(figsize=(10, 5))
    arm_df = df[df["Representation"] == "ARM"].copy()
    
    arm_melt = pd.melt(
        arm_df,
        id_vars=["Dataset_Clean"],
        value_vars=["Human_F1", "CV_F1"],
        var_name="Evaluator",
        value_name="F1_Score"
    )
    arm_melt["Evaluator"] = arm_melt["Evaluator"].map({"Human_F1": "Human Visual Inspection", "CV_F1": "CV Model (RetinaNet)"})

    ax = sns.barplot(
        data=arm_melt,
        x="Dataset_Clean",
        y="F1_Score",
        hue="Evaluator",
        palette=["#2ca02c", "#9467bd"]
    )
    plt.title("Human Visual Inspection vs. CV Model F1-Score across Custom Datasets (ARM)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Custom Dataset Category", fontsize=12, fontweight="bold")
    plt.ylabel("F1-Score", fontsize=12, fontweight="bold")
    plt.ylim(0.0, 1.35)
    plt.xticks(rotation=15)
    plt.legend(title="Evaluator", frameon=True, loc="upper center", ncol=2)

    for p in ax.patches:
        height = p.get_height()
        if height >= 0:
            ax.annotate(f"{height:.2f}",
                        (p.get_x() + p.get_width() / 2., height + 0.02),
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    pdf_out = os.path.join(fig_dir, "custom_eval_human_vs_cv_arm.pdf")
    png_out = os.path.join(fig_dir, "custom_eval_human_vs_cv_arm.png")
    plt.savefig(pdf_out, dpi=300, bbox_inches='tight')
    plt.savefig(png_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {pdf_out}")

    # 2. Figure 2: CV Model Performance Across Representations on Custom Datasets
    plt.figure(figsize=(11, 5.5))
    ax = sns.barplot(
        data=df,
        x="Dataset_Clean",
        y="CV_F1",
        hue="Representation_Clean",
        palette=color_map
    )
    plt.title("CV Model (RetinaNet) Concept Drift F1-Score across Representations on Custom Datasets", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Custom Dataset Category", fontsize=12, fontweight="bold")
    plt.ylabel("CV Model F1-Score", fontsize=12, fontweight="bold")
    plt.ylim(0.0, 1.35)
    plt.xticks(rotation=15)
    plt.legend(title="Representation", frameon=True, loc="upper center", ncol=3)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.2f}",
                        (p.get_x() + p.get_width() / 2., height + 0.02),
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    pdf_out = os.path.join(fig_dir, "custom_eval_cv_model_representations.pdf")
    png_out = os.path.join(fig_dir, "custom_eval_cv_model_representations.png")
    plt.savefig(pdf_out, dpi=300, bbox_inches='tight')
    plt.savefig(png_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {pdf_out}")

    # 3. Figure 3: Human Visual Inspection F1-Score Across Representations on Custom Datasets
    plt.figure(figsize=(11, 5.5))
    ax = sns.barplot(
        data=df,
        x="Dataset_Clean",
        y="Human_F1",
        hue="Representation_Clean",
        palette=color_map
    )
    plt.title("Human Visual Detectability F1-Score across Representations on Custom Datasets", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Custom Dataset Category", fontsize=12, fontweight="bold")
    plt.ylabel("Human Visual F1-Score", fontsize=12, fontweight="bold")
    plt.ylim(0.0, 1.35)
    plt.xticks(rotation=15)
    plt.legend(title="Representation", frameon=True, loc="upper center", ncol=3)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.2f}",
                        (p.get_x() + p.get_width() / 2., height + 0.02),
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    pdf_out = os.path.join(fig_dir, "custom_eval_human_representations.pdf")
    png_out = os.path.join(fig_dir, "custom_eval_human_representations.png")
    plt.savefig(pdf_out, dpi=300, bbox_inches='tight')
    plt.savefig(png_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {pdf_out}")


def generate_plots_and_tables():
    generate_benchmark_plots()
    generate_custom_eval_plots()


if __name__ == "__main__":
    generate_plots_and_tables()