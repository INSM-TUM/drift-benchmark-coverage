#!/bin/bash
#SBATCH -J cv4cdd_train
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o train_%j.log

module purge

# Navigate to the workspace root directory
cd "/storage/home/deig/nobackup/ARM-based-CV4CDD_backup"

# Activate virtual environment
source /storage/home/deig/.cache/pypoetry/virtualenvs/supervised-cd-cuda-AoR8uhDr-py3.9/bin/activate

# Load CUDA toolkit and libraries
module load cuda/11.8.0
export LD_LIBRARY_PATH=$HOME/local/cudnn/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 1. Resolve Experiment Config and Name
BASE_REP=${1:-arm}
experiment_name=$(python -c "import sys; sys.path.append('.'); import src.config as cfg; cfg.recalculate_derived_configs('$BASE_REP'); print(cfg.experiment_name)")
base_representation=$BASE_REP

if [ -z "$experiment_name" ]; then
    echo "Error: Could not extract experiment_name from src/config.py"
    exit 1
fi
echo "Experiment name resolved to: $experiment_name"
echo "Base representation resolved to: $base_representation"

# Create outputs folder if it doesn't exist
mkdir -p outputs

# Check if experiment is already fully completed
if [ -e "outputs/${experiment_name}_latest/completed.txt" ]; then
    echo "Experiment $experiment_name is already fully completed (completed.txt found). Exiting successfully."
    exit 0
fi

# 2. Resolve Isolated Run Directory and Resumption Status
RUN_DIR=""
RESUME_FLAG=""

# Check if resume is requested (either via environment variable or Slurm restart count)
if [ "$RESUME" = "true" ] || { [ -n "$SLURM_RESTART_COUNT" ] && [ "$SLURM_RESTART_COUNT" -gt 0 ]; }; then
    if [ -e "outputs/${experiment_name}_latest" ]; then
        RUN_DIR=$(readlink -f "outputs/${experiment_name}_latest")
        RESUME_FLAG="--resume"
        echo "Resuming training in existing run directory: $RUN_DIR"
    else
        echo "Warning: Resume requested but outputs/${experiment_name}_latest link not found. Starting a fresh run."
    fi
fi

if [ -z "$RUN_DIR" ]; then
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    RUN_DIR="/storage/home/deig/nobackup/ARM-based-CV4CDD_backup/outputs/${experiment_name}_${TIMESTAMP}"
    echo "Starting a fresh experiment run in: $RUN_DIR"
    mkdir -p "$RUN_DIR"
    ln -sfn "$RUN_DIR" "outputs/${experiment_name}_latest"
fi

# If DFG mode, run preprocessing for 'test' split only and exit
if [ "$base_representation" = "dfg" ]; then
    if [ -d "$RUN_DIR/preprocessed_data/test/winsim" ] && [ "$(ls -A $RUN_DIR/preprocessed_data/test/winsim 2>/dev/null)" ]; then
        echo -e "\n=================== DFG Preprocessing Already Completed ==================="
        touch "$RUN_DIR/completed.txt"
        exit 0
    fi
    echo -e "\n=================== DFG Representation Mode: Preprocessing Test Split Only ==================="
    python scripts/python/preprocess_pipeline.py --base_rep "$BASE_REP"
    if [ $? -eq 0 ]; then
        echo "DFG preprocessing completed. Writing completed.txt."
        touch "$RUN_DIR/completed.txt"
        exit 0
    else
        echo "Error in DFG preprocessing. Exiting."
        exit 1
    fi
fi

# 3. Phase A: Preprocessing (Skip if already completed)
if [ -f "$RUN_DIR/preprocessed_data/train/model_4d_cd-00000-of-00001.tfrecord" ] && [ -f "$RUN_DIR/preprocessed_data/val/model_4d_cd-00000-of-00001.tfrecord" ]; then
    echo -e "\n=================== Phase A: Skipping Preprocessing (.tfrecords already exist) ==================="
else
    echo -e "\n=================== Phase A: Preprocessing Event Logs (train and val) ==================="
    python scripts/python/preprocess_pipeline.py --base_rep "$BASE_REP"
    if [ $? -ne 0 ]; then
        echo "Error in Phase A (Preprocessing). Exiting."
        exit 1
    fi
fi

# 4. Phase B: Model Training (Skip if model already exported)
if [ -f "$RUN_DIR/exported_model/saved_model.pb" ]; then
    echo -e "\n=================== Phase B: Skipping Training (Model already exported) ==================="
else
    echo -e "\n=================== Phase B: Training the Model ==================="
    python scripts/python/train.py $RESUME_FLAG --base_rep "$BASE_REP"
    if [ $? -ne 0 ]; then
        echo "Error in Phase B (Training). Exiting."
        exit 1
    fi
fi

# Mark the run as fully completed
touch "$RUN_DIR/completed.txt"
echo -e "\nCV4CDD Pipeline Execution Finished Successfully."