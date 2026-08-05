#!/bin/bash
#SBATCH -J sensitivity_analysis
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o eval_cdrift_%j.log

set -e

# Configuration
DATASET="cdlg"     # Set to "cdlg" or "cdrift"
BASE_REP="dfg"     # Set to "arm", "dfg", or "dfg_io"

# Define window sizes to test
WINDOWS=(50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230)

CSV_FILE="window_sensitivity_${DATASET}_${BASE_REP}.csv"

echo "=========================================================="
echo "Running Window Sensitivity Analysis for $DATASET"
echo "Base Representation: $BASE_REP"
echo "Testing Window Sizes: ${WINDOWS[*]}"
echo "Output CSV: $CSV_FILE"
echo "=========================================================="

echo "window_size,lag,Precision,Recall,F1" > $CSV_FILE

for W in "${WINDOWS[@]}"; do
    echo ""
    echo "========================================"
    echo "Evaluating N_WINDOWS = $W"
    echo "========================================"
    
    if [ "$DATASET" == "cdlg" ]; then
        echo "Preprocessing CDLG Test Set with N_WINDOWS = $W..."
        poetry run python scripts/python/preprocess_pipeline.py --base_rep $BASE_REP --n_windows $W --input_dir data/input_cdlg/test --output_dir outputs/exp_${BASE_REP}_latest/preprocessed_data/test
        
        echo "Evaluating CDLG with N_WINDOWS = $W..."
        poetry run python scripts/python/evaluate_cdlg.py --base_rep $BASE_REP --n_windows $W
        
        # The output path depends on the window size
        OUT_CSV=$(find outputs/exp_${BASE_REP}_latest/exported_model/evaluation_w${W}/ -name "global_metrics.csv" | head -n 1)
        
    elif [ "$DATASET" == "cdrift" ]; then
        echo "Preprocessing CDrift Dataset with N_WINDOWS = $W..."
        poetry run python scripts/python/preprocess_pipeline.py --base_rep $BASE_REP --n_windows $W --input_dir data/input_cdrift/inscope --output_dir outputs/exp_${BASE_REP}_latest/cdrift_eval/preprocessed
        
        echo "Evaluating CDrift with N_WINDOWS = $W..."
        poetry run python scripts/python/evaluate_cdrift.py --base_rep $BASE_REP --n_windows $W
        
        OUT_CSV="outputs/exp_${BASE_REP}_latest/cdrift_evaluation/global_metrics.csv"
        
    else
        echo "Unknown dataset: $DATASET. Please use 'cdlg' or 'cdrift'."
        exit 1
    fi
    
    if [ -f "$OUT_CSV" ]; then
        echo "Extracting results from $OUT_CSV"
        # Extract row where lag is 0.05 (commonly used). If no 0.05 exists, we take the first available.
        # Format of global_metrics.csv: lag,Precision,Recall,F1,TP,FP,FN,n_logs
        # We prefix it with the window size.
        awk -F',' -v w="$W" '
            NR==2 && !found { fallback = w","$1","$2","$3","$4 }
            $1 == "0.05" { print w","$1","$2","$3","$4; found=1 }
            END { if(!found && fallback) print fallback }
        ' "$OUT_CSV" >> $CSV_FILE
    else
        echo "ERROR: Output CSV not found for N_WINDOWS=$W at expected location."
    fi
done

echo ""
echo "========================================"
echo "Evaluations complete! Generating plot..."
echo "========================================"

poetry run python scripts/python/plot_sensitivity.py $CSV_FILE

echo "Sensitivity analysis successfully finished!"
echo "Results saved to: $CSV_FILE"
echo "Plot saved to: ${CSV_FILE/.csv/.png}"
