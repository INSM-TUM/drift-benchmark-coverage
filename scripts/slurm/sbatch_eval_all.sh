#!/bin/bash
# Submits all evaluation jobs to the SLURM cluster

# Ensure we are in the correct directory (project root)
cd "$(dirname "$0")/../.."

MODELS=("arm" "dfg" "dfg_io")
EVAL_SCRIPTS=("sbatch_eval_cdrift.sh" "sbatch_eval_cdlg.sh" "sbatch_eval_custom.sh")

for model in "${MODELS[@]}"; do
    for script in "${EVAL_SCRIPTS[@]}"; do
        echo "Submitting $script for model $model..."
        sbatch "scripts/slurm/$script" "$model"
        # Small sleep to prevent queueing too fast and messing up log outputs
        sleep 1
    done
done

echo "All evaluation jobs queued successfully!"
