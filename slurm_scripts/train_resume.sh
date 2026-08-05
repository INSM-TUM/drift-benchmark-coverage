#!/bin/bash
# train_resume.sh
# SLURM worker script to run or resume training for an object detection model.
#
# Usage:
#   sbatch train_resume.sh [MODEL_DIR] [OUTPUT_DIR] [TRAIN_DATA_DIR] [EVAL_DATA_DIR] [REMAINING_CHAIN_COUNT]
#
# All parameters have default mock server paths if not specified.

#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=45
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --signal=B:USR1@120

set -euo pipefail

# Command-line or environment default arguments (mock paths on remote server)
MODEL_DIR="${1:-/storage/home/deig/nobackup/cv4cdd/data/model_training_logging/experiment_resume}"
OUTPUT_DIR="${2:-/storage/home/deig/nobackup/cv4cdd/data/output/exported_model}"
TRAIN_DATA_DIR="${3:-/storage/home/deig/nobackup/cv4cdd/data/tf_records/train/model_4d-00000-of-00001.tfrecord}"
EVAL_DATA_DIR="${4:-/storage/home/deig/nobackup/cv4cdd/data/tf_records/val/model_4d-00000-of-00001.tfrecord}"
REMAINING_CHAIN_COUNT="${5:-3}"

# Ensure logs directory exists
mkdir -p "/storage/home/deig/nobackup/cv4cdd/slurm_scripts/logs"

echo "=============================================================="
echo "CV4CDD Slurm Resume Training Job"
echo "Date:                      $(date)"
echo "Host:                      $(hostname)"
echo "SLURM Job ID:              ${SLURM_JOB_ID:-N/A}"
echo "Model Checkpoint Dir:      $MODEL_DIR"
echo "Export Output Dir:         $OUTPUT_DIR"
echo "Train TFRecord Path:       $TRAIN_DATA_DIR"
echo "Eval TFRecord Path:        $EVAL_DATA_DIR"
echo "Remaining Chain Count:     $REMAINING_CHAIN_COUNT"
echo "=============================================================="

# Fast exit if training completed in a previous job run
COMPLETED_MARKER="${MODEL_DIR}/TRAINING_COMPLETED"
if [[ -f "$COMPLETED_MARKER" ]]; then
    echo "[INFO] Marker file $COMPLETED_MARKER already exists."
    echo "[INFO] Training is already finished for this experiment. Exiting cleanly."
    exit 0
fi

# Load modules & virtual environment
module purge
cd "/storage/home/deig/nobackup/cv4cdd/approaches/object_detection" || { echo "Failed to navigate to object detection dir"; exit 1; }

source /storage/home/deig/.cache/pypoetry/virtualenvs/supervised-cd-cuda-AoR8uhDr-py3.9/bin/activate
module load cuda/11.8.0
export LD_LIBRARY_PATH=$HOME/local/cudnn/lib64:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# Environment performance optimizations
export OMP_NUM_THREADS=45
export TF_NUM_INTRAOP_THREADS=45
export TF_NUM_INTEROP_THREADS=4
export TF_CPP_MIN_LOG_LEVEL=1

# Execute Python training script (resumes automatically if checkpoints exist)
set +e
python resume_training.py \
    --model_dir "$MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --train_data_dir "$TRAIN_DATA_DIR" \
    --eval_data_dir "$EVAL_DATA_DIR"
PYTHON_EXIT_CODE=$?
set -e

echo "--------------------------------------------------------------"
echo "Python execution completed with exit code: $PYTHON_EXIT_CODE"
echo "--------------------------------------------------------------"

# Handle auto-resubmission if job timed out before reaching final step count
if [[ -f "$COMPLETED_MARKER" ]]; then
    echo "=============================================================="
    echo "[SUCCESS] Training reached target steps and created $COMPLETED_MARKER."
    echo "=============================================================="
else
    if [[ "$REMAINING_CHAIN_COUNT" -gt 0 ]]; then
        NEXT_CHAIN=$((REMAINING_CHAIN_COUNT - 1))
        echo "=============================================================="
        echo "[INFO] Training is not complete yet. Resubmitting job to Slurm..."
        echo "[INFO] Next remaining chain count: $NEXT_CHAIN"
        echo "=============================================================="
        
        cd "/storage/home/deig/nobackup/cv4cdd/slurm_scripts"
        sbatch \
            -J "resume_train" \
            -o "logs/train_resume_%j.log" \
            train_resume.sh "$MODEL_DIR" "$OUTPUT_DIR" "$TRAIN_DATA_DIR" "$EVAL_DATA_DIR" "$NEXT_CHAIN"
    else
        echo "[WARNING] Reached maximum resubmission chain limit ($REMAINING_CHAIN_COUNT). Stop resubmitting."
    fi
fi
