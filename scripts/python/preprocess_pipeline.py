import os
import sys
import argparse

# Add root directory to path to allow importing src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import src.config as cfg
import src.preprocessing as pp

def process_standard_splits(run_dir):
    """Automated preprocessing mode (processes CDLG train/val/test splits)."""
    print(f"Running automated standard preprocessing. Saving outputs to: {run_dir}")
    
    splits = ["train", "val", "test"]
    if getattr(cfg, "BASE_REPRESENTATION", "arm") == "dfg":
        splits = ["test"]
        print("BASE_REPRESENTATION is 'dfg'. Only processing 'test' split.")
        
    for split in splits:
        print(f"\n================ Processing Split: {split.upper()} ================")
        
        log_dir = getattr(cfg, f"CDLG_{split.upper()}_DIR")
        if not os.path.exists(log_dir):
            print(f"Warning: Log directory {log_dir} does not exist. Skipping split '{split}'.")
            continue
            
        output_split_dir = os.path.join(run_dir, "preprocessed_data", split)
        
        print(f"Starting generic preprocessing pipeline for {split} split...")
        pp.process_generic_directory(
            input_dir=log_dir,
            output_dir=output_split_dir,
            create_tfrecords=True
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser("preprocess_pipeline")
    parser.add_argument("--base_rep", dest="base_rep",
                        help="Base representation (e.g., 'arm' or 'dfg')",
                        default="arm",
                        type=str)
    # Generic Mode Arguments
    parser.add_argument("--input_dir", dest="input_dir", help="Path to input directory of .xes files (for manual generic mode).", type=str)
    parser.add_argument("--output_dir", dest="output_dir", help="Path to output directory (for manual generic mode).", type=str)
    parser.add_argument("--tfrecords", dest="tfrecords", action="store_true", help="Whether to generate .tfrecords from images.")
    parser.add_argument("--n_windows", dest="n_windows", help="Override N_WINDOWS at runtime (avoids modifying config.py).", type=int, default=None)
    
    args = parser.parse_args()
    
    # Recalculate config using the specified base_rep
    cfg.recalculate_derived_configs(args.base_rep)
    
    if args.n_windows is not None:
        cfg.N_WINDOWS = args.n_windows
        print(f"Overriding N_WINDOWS = {cfg.N_WINDOWS}")
    
    # Mode B: Manual Generic processing of an arbitrary folder
    if args.input_dir and args.output_dir:
        print(f"Manual processing mode triggered for: {args.input_dir}")
        pp.process_generic_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            create_tfrecords=args.tfrecords
        )
    # Mode A: Automated processing of standard CDLG splits
    else:
        # Get run_dir directly from config since we don't pass it anymore!
        run_dir = os.path.join(cfg.ROOT_DIR, f"outputs/{cfg.experiment_name}_latest")
        process_standard_splits(run_dir)
