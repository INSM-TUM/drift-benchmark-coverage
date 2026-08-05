import tensorflow_models as tfm
import tensorflow as tf
import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import src.config as cfg
import src.utilities as utils

from official.vision.serving import export_saved_model_lib
from typing import Tuple

def find_latest_checkpoint_dir(parent_dir):
    if not os.path.exists(parent_dir):
        return None
    
    subdirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) 
               if os.path.isdir(os.path.join(parent_dir, d))]
    
    checkpoint_subdirs = []
    for d in subdirs:
        has_checkpoint = False
        try:
            for f in os.listdir(d):
                if f == "checkpoint" or f.startswith("ckpt-") or "model.ckpt" in f:
                    has_checkpoint = True
                    break
        except OSError:
            continue
        if has_checkpoint:
            checkpoint_subdirs.append(d)
            
    if not checkpoint_subdirs:
        return None
        
    return max(checkpoint_subdirs, key=os.path.getmtime)

def train(model_dir=cfg.MODEL_PATH, 
          output_dir=cfg.DEFAULT_OUTPUT_DIR,
          resume: bool = False) -> Tuple[tf.keras.Model, dict]:
    """Main function for training the object detection models.

    Args:
        model_dir (str, optional): TensorFlow Model path. Defaults to cfg.MODEL_PATH.
        output_dir (str, optional): Output directory. 
            Defaults to cfg.DEFAULT_OUTPUT_DIR.
        resume (bool, optional): Whether to resume from an existing checkpoint.
    """
    actual_model_dir = model_dir
    has_chkpt_directly = False
    if os.path.exists(model_dir):
        try:
            for f in os.listdir(model_dir):
                if f == "checkpoint" or f.startswith("ckpt-") or "model.ckpt" in f:
                    has_chkpt_directly = True
                    break
        except OSError:
            pass
            
    if resume:
        if has_chkpt_directly:
            # If the checkpoint is directly in the directory, resolve the output directory
            parent_dir = os.path.dirname(model_dir)
            if os.path.basename(model_dir) == "checkpoints":
                output_dir = os.path.join(parent_dir, "exported_model")
            elif output_dir == cfg.DEFAULT_OUTPUT_DIR:
                run_name = os.path.basename(model_dir)
                output_dir = os.path.join(output_dir, run_name)
        else:
            latest_dir = find_latest_checkpoint_dir(model_dir)
            if latest_dir:
                actual_model_dir = latest_dir
                print(f"Automatically selected latest checkpoint directory for resume: {actual_model_dir}")
                if output_dir == cfg.DEFAULT_OUTPUT_DIR:
                    # Resolve to parent output dir if it contains checkpoints/
                    parent_run = os.path.dirname(actual_model_dir)
                    if os.path.basename(actual_model_dir) == "checkpoints":
                        output_dir = os.path.join(parent_run, "exported_model")
                    else:
                        run_name = os.path.basename(actual_model_dir)
                        output_dir = os.path.join(output_dir, run_name)
            else:
                print(f"No existing checkpoints found in {model_dir} or its subdirectories. Starting a fresh training run.")
                resume = False

    if not resume:
        timestamp = utils.get_timestamp()
        # If we are using a structured run directory, keep the checkpoints/ structure
        if os.path.basename(model_dir) == "checkpoints":
            # Keep model_dir and output_dir as they are (preset in args.run_dir logic)
            pass
        else:
            actual_model_dir = os.path.join(model_dir,
                                     f"{timestamp}_{cfg.ENCODING_TYPE}_{cfg.OPTIMIZER_TYPE}")
            output_dir = os.path.join(output_dir,
                                      f"{timestamp}_{cfg.ENCODING_TYPE}_{cfg.OPTIMIZER_TYPE}")

    exp_config = utils.get_model_config(actual_model_dir)

    if exp_config.runtime.mixed_precision_dtype == tf.float16:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    logical_device_names = [
        logical_device.name for logical_device in tf.config.list_logical_devices()]

    if 'GPU' in ''.join(logical_device_names):
        distribution_strategy = tf.distribute.MirroredStrategy()
    else:
        print('Warning: this will be really slow. Using CPU')
        distribution_strategy = tf.distribute.OneDeviceStrategy(
            logical_device_names[0])

    print('Setup Done')

    with distribution_strategy.scope():
        task = tfm.core.task_factory.get_task(
            exp_config.task, logging_dir=actual_model_dir)

    for images, labels in task.build_inputs(exp_config.task.train_data).take(1):
        print()
        print(
            f'images.shape: {str(images.shape):16}  images.dtype: {images.dtype!r}')
        print(f'labels.keys: {labels.keys()}')

    model, eval_logs = tfm.core.train_lib.run_experiment(
        distribution_strategy=distribution_strategy,
        task=task,
        mode='train_and_eval',
        params=exp_config,
        model_dir=actual_model_dir,
        run_post_eval=True)

    save_options = tf.saved_model.SaveOptions(experimental_custom_gradients=True)

    export_saved_model_lib.export_inference_graph(
        input_type='image_tensor',
        batch_size=1,
        input_image_size=[cfg.HEIGHT, cfg.WIDTH],
        params=exp_config,
        log_model_flops_and_params=True,
        save_options=save_options,
        checkpoint_path=tf.train.latest_checkpoint(actual_model_dir),
        export_dir=output_dir)

    return model, eval_logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--gpu_devices", dest="gpu_devices",
                        help="Specify which CUDA devices to use.",
                        default="",
                        type=str)
    parser.add_argument("--model_dir", dest="model_dir",
                        help="Path to the model directory for training or resume.",
                        default=cfg.MODEL_PATH,
                        type=str)
    parser.add_argument("--output_dir", dest="output_dir",
                        help="Path to the output export directory.",
                        default=cfg.DEFAULT_OUTPUT_DIR,
                        type=str)
    parser.add_argument("--resume", dest="resume",
                        help="Resume training from an existing checkpoint directory.",
                        action="store_true")
    parser.add_argument("--base_rep", dest="base_rep",
                        help="Base representation (e.g., 'arm' or 'dfg')",
                        default="arm",
                        type=str)
    args = parser.parse_args()

    cfg.recalculate_derived_configs(args.base_rep)
    run_dir = os.path.join(cfg.ROOT_DIR, f"outputs/{cfg.experiment_name}_latest")
    
    # Override paths relative to the run directory
    cfg.MODEL_PATH = os.path.join(run_dir, "checkpoints")
    cfg.DEFAULT_OUTPUT_DIR = os.path.join(run_dir, "exported_model")
    cfg.TRAIN_DATA_DIR = os.path.join(run_dir, "preprocessed_data/train/model_4d_cd-00000-of-00001.tfrecord")
    cfg.EVAL_DATA_DIR = os.path.join(run_dir, "preprocessed_data/val/model_4d_cd-00000-of-00001.tfrecord")

    args.model_dir = cfg.MODEL_PATH
    args.output_dir = cfg.DEFAULT_OUTPUT_DIR

    # set cuda devices
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    # Check command line resume flag or config option
    resume_flag = args.resume or getattr(cfg, "RESUME", False)

    model, eval_logs = train(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        resume=resume_flag)
