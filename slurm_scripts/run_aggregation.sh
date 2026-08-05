#!/bin/bash
#SBATCH -J cv4cdd_evaluate
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o cdrift_evaluate_%j.log

set -euo pipefail

module purge

# Navigate to the workspace root directory
cd "/storage/home/deig/nobackup/cv4cdd/approaches/object_detection" || { echo "Failed to change directory to /storage/home/deig/nobackup/CV4CDD/approaches/object_detection"; exit 1; }

# Activate virtual environment
source /storage/home/deig/.cache/pypoetry/virtualenvs/supervised-cd-cuda-AoR8uhDr-py3.9/bin/activate

# Load CUDA toolkit and libraries
module load cuda/11.8.0
export LD_LIBRARY_PATH=$HOME/local/cudnn/lib64:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# Optimize multi-threading for TensorFlow and NumPy
export OMP_NUM_THREADS=64
export TF_NUM_INTRAOP_THREADS=64
export TF_NUM_INTEROP_THREADS=4

# Reduce TensorFlow verbosity (show errors and warnings only)
export TF_CPP_MIN_LOG_LEVEL=1

echo "=============================================================="
echo "CV4CDD Custom Datasets Evaluation"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "SLURM Job ID: ${SLURM_JOB_ID:-N/A}"
echo "=============================================================="

BASE_REP=${1:-arm}

# Run evaluate_custom.py
python aggregate_cdlg_results.py

echo "Evaluation completed successfully."
