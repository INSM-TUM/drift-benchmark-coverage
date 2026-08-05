import os
import random
import tensorflow as tf

import utils.config as cfg
import utils.utilities as utils
import utils.evaluation as eval


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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("evaluate")
    parser.add_argument("--model-path", dest="model_path",
                        help="Specify which model to use.",
                        default=cfg.TRAINED_MODEL_PATH,
                        type=str)
    parser.add_argument("--data-dir", dest="data_dir",
                        help="Specify directory of evaluation data.",
                        default=cfg.TEST_IMAGE_DATA_DIR,
                        type=str)
    parser.add_argument("--base-rep", dest="base_rep",
                        help="Specify the base representation to use.",
                        default="dfg",
                        choices=["dfg", "dfg_io", "arm"],
                        required=False,
                        type=str)
    args = parser.parse_args()

    cfg.BASE_REPRESENTATION = args.base_rep

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    model = tf.saved_model.load(args.model_path)
    # visualize_bboxes(model) # Optional, can be uncommented if visualizations are needed

    eval.evaluate(data_dir=args.data_dir,
                  model=model,
                  threshold=cfg.EVAL_THRESHOLD)
