#!/bin/bash
# run_resume_training_job.sh
# Master launcher script for submitting resume-enabled training jobs to Slurm.
#
# Usage:
#   ./run_resume_training_job.sh [MODEL_DIR] [OUTPUT_DIR] [TRAIN_DATA_DIR] [EVAL_DATA_DIR] [MAX_RESUBMISSIONS]
#
# Example:
#   ./run_resume_training_job.sh /path/to/models /path/to/output /path/to/train.tfrecord /path/to/val.tfrecord 4

set -euo pipefail

MODEL_DIR="${1:-/storage/home/deig/nobackup/cv4cdd/data/model_training_logging/experiment_resume}"
OUTPUT_DIR="${2:-/storage/home/deig/nobackup/cv4cdd/data/output/exported_model}"
TRAIN_DATA_DIR="${3:-/storage/home/deig/nobackup/cv4cdd/data/tf_records/train/model_4d-00000-of-00001.tfrecord}"
EVAL_DATA_DIR="${4:-/storage/home/deig/nobackup/cv4cdd/data/tf_records/val/model_4d-00000-of-00001.tfrecord}"
MAX_RESUBMISSIONS="${5:-3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

echo "=============================================================="
echo "Submitting Resume Training Job to Slurm"
echo "Model Dir:       $MODEL_DIR"
echo "Output Dir:      $OUTPUT_DIR"
echo "Train Data:      $TRAIN_DATA_DIR"
echo "Eval Data:       $EVAL_DATA_DIR"
echo "Max Resubmit:    $MAX_RESUBMISSIONS"
echo "=============================================================="

JOB_ID=$(sbatch \
    --parsable \
    -J "resume_train" \
    -o "logs/train_resume_%j.log" \
    train_resume.sh "$MODEL_DIR" "$OUTPUT_DIR" "$TRAIN_DATA_DIR" "$EVAL_DATA_DIR" "$MAX_RESUBMISSIONS")

echo "Job submitted successfully! Initial Slurm Job ID: $JOB_ID"
echo "Monitor log with: tail -f logs/train_resume_${JOB_ID}.log"
