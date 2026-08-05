import os
import pandas as pd

import utils.preprocessing as pp
import utils.config as cfg
import utils.utilities as seg_utils
import utils.vdd_helper as vdd

def generate_only_annotations(data_dir: str):
    """Generate only annotations for given data

    Args:
        data_dir (str): Image data directory
    """
    drift_info = pd.read_csv(os.path.join(data_dir, "drift_info.csv"), sep=";")
    log_matching = pd.read_csv(os.path.join(data_dir, "log_matching.csv"))
    
    log_matching = dict(log_matching.values)
    
    log_names = log_matching.keys()
    
    if cfg.VDD_PREPROCESSING:
        print("Generating VDD annotations")
        vdd.generate_vdd_annotations(drift_info=drift_info,
                                     dir=data_dir,
                                     log_matching=log_matching,
                                     log_names=log_names)
    else:
        print("Generating WINSIM annotations")
        seg_utils.generate_annotations(drift_info=drift_info,
                                dir=data_dir,
                                log_matching=log_matching,
                                log_names=log_names)
    if cfg.AUTOMATE_TFR_SCRIPT:
        seg_utils.start_tfr_script(repo_dir=cfg.TENSORFLOW_MODELS_DIR,
                                data_dir=cfg.DEFAULT_DATA_DIR,
                                tfr_dir=cfg.TFR_RECORDS_DIR,
                                prefix=cfg.OUTPUT_PREFIX)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Detection Pipeline")
    parser.add_argument("--base-rep", dest="base_rep",
                        help="Specify the base representation: dfg, dfg_io, arm",
                        default=None,
                        choices=["dfg", "dfg_io", "arm"],
                        required=False,
                        type=str)
    parser.add_argument("--data-dir", dest="data_dir", type=str, default=None, help="Override DEFAULT_DATA_DIR")
    parser.add_argument("--log-dir", dest="log_dir", type=str, default=None, help="Override DEFAULT_LOG_DIR")
    args = parser.parse_args()

    if args.base_rep:
        cfg.BASE_REPRESENTATION = args.base_rep
    if args.data_dir:
        cfg.DEFAULT_DATA_DIR = args.data_dir
    if args.log_dir:
        cfg.DEFAULT_LOG_DIR = args.log_dir

    if cfg.ANNOTATIONS_ONLY:
        generate_only_annotations(cfg.DEFAULT_DATA_DIR)
    elif cfg.VDD_PREPROCESSING:
        print("Starting VDD pipeline")
        pp.vdd_pipeline()
    else:
        print("Starting WINSIM pipeline")
        pp.winsim_pipeline(n_windows=cfg.N_WINDOWS)
    
    
