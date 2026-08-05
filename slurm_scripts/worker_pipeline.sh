#!/bin/bash
# worker_pipeline.sh
# SLURM script to run the prediction and evaluation pipeline for a specific Dataset + Representation
#
# Usage (called by run_all_experiments.sh): sbatch worker_pipeline.sh <DATASET> <REP>
# Example: sbatch worker_pipeline.sh cdrift arm

#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=45
#SBATCH --mem=64G
#SBATCH -t 20:00:00

set -euo pipefail

# Argument Parsing
DATASET=$1
BASE_REP=$2

if [[ -z "$DATASET" || -z "$BASE_REP" ]]; then
    echo "Error: Must provide DATASET and BASE_REP arguments."
    exit 1
fi

module purge

# Navigate to the workspace root directory
cd "/storage/home/deig/nobackup/cv4cdd/approaches/object_detection" || { echo "Failed to change directory"; exit 1; }

# Activate virtual environment
source /storage/home/deig/.cache/pypoetry/virtualenvs/supervised-cd-cuda-AoR8uhDr-py3.9/bin/activate

# Load CUDA toolkit and libraries
module load cuda/11.8.0
export LD_LIBRARY_PATH=$HOME/local/cudnn/lib64:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# Optimize multi-threading for TensorFlow and NumPy
export OMP_NUM_THREADS=45
export TF_NUM_INTRAOP_THREADS=45
export TF_NUM_INTEROP_THREADS=4

# Reduce TensorFlow verbosity (show errors and warnings only)
export TF_CPP_MIN_LOG_LEVEL=1

echo "=============================================================="
echo "CV4CDD Pipeline: Dataset=$DATASET | Base Representation=$BASE_REP"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "SLURM Job ID: ${SLURM_JOB_ID:-N/A}"
echo "=============================================================="

# Common Configuration
MODEL_PATH="/storage/home/deig/nobackup/ARM-based-CV4CDD_backup/outputs/exp_arm_latest/exported_model"

if [[ "$DATASET" == "cdrift" ]]; then
    
    LOG_DIR="/storage/home/deig/nobackup/ARM-based-CV4CDD_backup/data/input_cdrift/inscope"
    OUTPUT_BASE_DIR="/storage/home/deig/nobackup/cv4cdd/data/cdrift_predictions_arm_${BASE_REP}"
    
    # 1. Predictions: cdrift evaluates across window sizes 60 to 200 in steps of 10
    WINDOWS=(60 70 80 90 100 110 120 130 140 150 160 170 180 190 200)
    
    for W in "${WINDOWS[@]}"; do
        PRED_FILE="${OUTPUT_BASE_DIR}/winsim_${W}/prediction_results.csv"
        if [[ -f "$PRED_FILE" ]]; then
            echo "Prediction file $PRED_FILE already exists. Skipping prediction for window size $W."
        else
            echo "Running Prediction for cdrift with window size $W..."
            python predict.py \
                --model-path "$MODEL_PATH" \
                --log-dir "$LOG_DIR" \
                --encoding winsim \
                --n-windows "$W" \
                --base-rep "$BASE_REP" \
                --output-dir "${OUTPUT_BASE_DIR}/winsim_${W}"
        fi
    done
    
    # 2. Evaluation across windows
    EVAL_ALL_FILE="${OUTPUT_BASE_DIR}/prediction_results_evaluated_all_windows.csv"
    if [[ -f "$EVAL_ALL_FILE" ]]; then
        echo "Evaluated file $EVAL_ALL_FILE already exists. Skipping evaluate_scdd_4d_on_external_data.py."
    else
        echo "Running Evaluation for cdrift..."
        python evaluate_scdd_4d_on_external_data.py \
            --base-path "$OUTPUT_BASE_DIR" \
            --num-cores 45
    fi

    # 3. Calculate total metrics
    TOTAL_METRICS_FILE="${OUTPUT_BASE_DIR}/total_metrics.txt"
    if [[ -f "$TOTAL_METRICS_FILE" ]]; then
        echo "Total metrics file $TOTAL_METRICS_FILE already exists. Skipping cdrift_calc_metrics.py."
    else
        echo "Running cdrift_calc_metrics.py..."
        python cdrift_calc_metrics.py \
            -p "$EVAL_ALL_FILE" \
            -l 0.05 \
            -w 70 \
            -o "$OUTPUT_BASE_DIR"
    fi

elif [[ "$DATASET" == "cdlg" ]]; then
    
    LOG_DIR="/storage/home/deig/nobackup/ARM-based-CV4CDD_backup/data/input_cdlg/test"
    OUTPUT_BASE_DIR="/storage/home/deig/nobackup/cv4cdd/data/cdlg_predictions_arm_model_${BASE_REP}"
    
    PRED_FILE="${OUTPUT_BASE_DIR}/prediction_results.csv"
    if [[ -f "$PRED_FILE" ]]; then
        echo "Prediction file $PRED_FILE already exists. Skipping prediction for cdlg."
    else
        echo "Running Prediction for cdlg..."
        python predict.py \
            --model-path "$MODEL_PATH" \
            --log-dir "$LOG_DIR" \
            --encoding winsim \
            --n-windows 200 \
            --base-rep "$BASE_REP" \
            --output-dir "${OUTPUT_BASE_DIR}"
    fi

    TOTAL_METRICS_FILE="${OUTPUT_BASE_DIR}/total_metrics.txt"
    if [[ -f "$TOTAL_METRICS_FILE" ]]; then
        echo "Total metrics file $TOTAL_METRICS_FILE already exists. Skipping calculate_cdlg_evaluation_results.py."
    else
        echo "Running Ground Truth Evaluation for cdlg..."
        python calculate_cdlg_evaluation_results.py \
            -p "$PRED_FILE" \
            -d "${LOG_DIR}/drift_info.csv" \
            -t "${LOG_DIR}/number_of_traces.json" \
            -l 0.05 \
            -o "${OUTPUT_BASE_DIR}"
    fi

elif [[ "$DATASET" == "custom_eval" ]]; then
    
    BASE_EVAL_DIR="/storage/home/deig/nobackup/ARM-based-CV4CDD_backup/data/custom_eval"
    OUTPUT_BASE_DIR="/storage/home/deig/nobackup/cv4cdd/data/custom_eval_predictions_arm_model_${BASE_REP}"
    
    SUBDIRS=("between_activities" "big" "gradual" "isolated" "loops")
    
    for SUBDIR in "${SUBDIRS[@]}"; do
        echo "Processing custom_eval subset: $SUBDIR..."
        
        LOG_DIR="${BASE_EVAL_DIR}/${SUBDIR}"
        SUBDIR_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${SUBDIR}"
        PRED_FILE="${SUBDIR_OUTPUT_DIR}/prediction_results.csv"
        TOTAL_METRICS_FILE="${SUBDIR_OUTPUT_DIR}/total_metrics.txt"

        if [[ -f "$PRED_FILE" ]]; then
            echo "  Prediction file $PRED_FILE already exists. Skipping prediction for $SUBDIR."
        else
            echo "  Running Prediction..."
            python predict.py \
                --model-path "$MODEL_PATH" \
                --log-dir "$LOG_DIR" \
                --encoding winsim \
                --n-windows 200 \
                --base-rep "$BASE_REP" \
                --output-dir "$SUBDIR_OUTPUT_DIR"
        fi

        if [[ -f "$TOTAL_METRICS_FILE" ]]; then
            echo "  Total metrics file $TOTAL_METRICS_FILE already exists. Skipping evaluation for $SUBDIR."
        else
            echo "  Running Ground Truth Evaluation for $SUBDIR..."
            python calculate_cdlg_evaluation_results.py \
                -p "$PRED_FILE" \
                -d "${LOG_DIR}/drift_info.csv" \
                -t "${LOG_DIR}/number_of_traces.json" \
                -l 0.05 \
                -o "$SUBDIR_OUTPUT_DIR"
        fi
    done

else
    echo "Unknown dataset: $DATASET"
    exit 1
fi

echo "=============================================================="
echo "Pipeline for $DATASET ($BASE_REP) completed successfully."
echo "=============================================================="
