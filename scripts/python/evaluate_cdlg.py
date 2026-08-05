import os
import sys
import random
import tensorflow as tf

# Add root directory to path to allow importing src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import src.config as cfg
import src.utilities as utils
import src.evaluation as eval


def visualize_bboxes(model: tf.keras.Model):
    """Call visualizing functions.

    Args:
        model (tf.keras.Model): TensorFlow model
    """
    seed = random.randint(0, 10000)

    utils.visualize_batch(path=cfg.EVAL_DATA_DIR,
                          mode="validation",
                          seed=seed)
    utils.visualize_predictions(path=cfg.EVAL_DATA_DIR,
                                mode="validation",
                                model=model,
                                seed=seed,
                                threshold=cfg.EVAL_THRESHOLD,
                                encoding="winsim")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("evaluate")
    parser.add_argument("--base_rep", dest="base_rep",
                        help="Base representation (e.g., 'arm' or 'dfg')",
                        default="arm",
                        type=str)
    parser.add_argument("--gpu_devices", dest="gpu_devices",
                        help="Specify which CUDA devices to use. Default is empty (CPU).",
                        default="",
                        type=str)
    parser.add_argument("--n_windows", dest="n_windows",
                        help="Override N_WINDOWS at runtime (avoids modifying config.py).",
                        default=None,
                        type=int)
    args = parser.parse_args()

    # set cuda devices
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    cfg.recalculate_derived_configs(args.base_rep)

    if args.n_windows is not None:
        cfg.N_WINDOWS = args.n_windows
        print(f"Overriding N_WINDOWS = {cfg.N_WINDOWS}")

    if getattr(cfg, 'AUTO_RUN_PREPROCESSING', False):
        import src.preprocessing as pp
        print(f"Auto-running preprocessing for CDLG test data from {cfg.CDLG_TEST_DIR}...")
        pp.process_generic_directory(
            input_dir=cfg.CDLG_TEST_DIR,
            output_dir=cfg.CDLG_TEST_OUTPUT_DIR,
            create_tfrecords=True
        )

    # Locate the nested winsim experiment directory created by preprocessing
    import glob
    winsim_dirs = glob.glob(os.path.join(cfg.TEST_IMAGE_DATA_DIR, f"experiment_*_w{cfg.N_WINDOWS}"))
    
    # Filter only fully completed experiments that have window_info.json
    valid_dirs = [d for d in winsim_dirs if os.path.isfile(os.path.join(d, "window_info.json"))]
    
    if valid_dirs:
        actual_preprocessed_dir = sorted(valid_dirs)[-1]
        print(f"Auto-resolved TEST_IMAGE_DATA_DIR to: {actual_preprocessed_dir}")
        cfg.TEST_IMAGE_DATA_DIR = actual_preprocessed_dir
    else:
        print(f"Warning: Could not find completed winsim experiment folder for w{cfg.N_WINDOWS} in {cfg.TEST_IMAGE_DATA_DIR}")

    raw_drift_info_path = cfg.CDLG_TEST_DRIFT_INFO
    if os.path.isfile(raw_drift_info_path):
        print(f"Auto-resolved raw_drift_info to: {raw_drift_info_path}")
    else:
        print(f"Warning: Default raw drift_info not found at {raw_drift_info_path}")
        raw_drift_info_path = None

    print(f"Loading exported model from: {cfg.TRAINED_MODEL_PATH}")
    model = tf.saved_model.load(cfg.TRAINED_MODEL_PATH)
    visualize_bboxes(model)

    print(f"Evaluating model on: {cfg.TEST_IMAGE_DATA_DIR}")
    eval.evaluate(data_dir=cfg.TEST_IMAGE_DATA_DIR,
                  model=model,
                  threshold=cfg.EVAL_THRESHOLD,
                  raw_drift_info_path=raw_drift_info_path)
