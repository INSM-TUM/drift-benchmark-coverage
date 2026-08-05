#!/bin/bash
# run_all_experiments.sh
# Master script to orchestrate the SLURM prediction and evaluation pipeline.
#
# Usage: ./run_all_experiments.sh

# Define the datasets and base representations to run
DATASETS=("cdrift")
REPRESENTATIONS=("arm") # ("dfg" "dfg_io" "arm")

cd "/storage/home/deig/nobackup/cv4cdd/slurm_scripts"

# Create a logs directory for the SLURM outputs
mkdir -p logs

for DATASET in "${DATASETS[@]}"; do
    for REP in "${REPRESENTATIONS[@]}"; do
        
        echo "Submitting job for Dataset: $DATASET | Representation: $REP"
        
        # Submit the worker script as a SLURM job, passing DATASET and REP as arguments
        # The job name and log file name are dynamically set based on the parameters
        sbatch \
            -J "${DATASET}_${REP}" \
            -o "logs/${DATASET}_${REP}_%j.log" \
            worker_pipeline_arm_model.sh "$DATASET" "$REP"
            
    done
done

echo "All 6 jobs have been submitted to SLURM!"
