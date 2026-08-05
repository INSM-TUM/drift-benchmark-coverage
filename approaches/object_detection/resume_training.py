import tensorflow_models as tfm
import tensorflow as tf
import os
import sys
import argparse
import signal
import logging
from typing import Tuple

import utils.config as cfg
import utils.utilities as utils

from official.vision.serving import export_saved_model_lib


def handle_slurm_signal(signum, frame):
    """Handle Slurm timeout signals (SIGUSR1 / SIGTERM) for graceful shutdown."""
    print(f"\n[INFO] Received signal {signum} (Slurm time limit warning). Flushing output buffers...")
    sys.stdout.flush()
    sys.stderr.flush()


def run_resume_training(
    model_dir: str,
    output_dir: str,
    train_data_dir: str,
    eval_data_dir: str,
    train_steps: int = None,
    train_batch_size: int = None,
    eval_batch_size: int = None,
    checkpoint_interval: int = None,
    learning_rate: float = None,
    force: bool = False
) -> Tuple[tf.keras.Model, dict]:
    """Train or resume object detection training from the latest checkpoint.

    Args:
        model_dir (str): Directory where model checkpoints and logs are saved.
        output_dir (str): Directory where exported saved model is saved.
        train_data_dir (str): Absolute path to training TFRecord file.
        eval_data_dir (str): Absolute path to validation TFRecord file.
        train_steps (int, optional): Total training steps. Defaults to cfg.TRAIN_STEPS.
        train_batch_size (int, optional): Global train batch size.
        eval_batch_size (int, optional): Global evaluation batch size.
        checkpoint_interval (int, optional): Steps between checkpoints.
        learning_rate (float, optional): Initial learning rate.
        force (bool, optional): Force retraining even if completed flag exists.

    Returns:
        Tuple[tf.keras.Model, dict]: Trained TensorFlow model and evaluation logs.
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    completed_file = os.path.join(model_dir, "TRAINING_COMPLETED")

    # Load experiment config template
    exp_config = utils.get_model_config(model_dir)

    # Apply overrides
    exp_config.task.train_data.input_path = train_data_dir
    exp_config.task.validation_data.input_path = eval_data_dir

    if train_steps is not None:
        exp_config.trainer.train_steps = train_steps
    if train_batch_size is not None:
        exp_config.task.train_data.global_batch_size = train_batch_size
    if eval_batch_size is not None:
        exp_config.task.validation_data.global_batch_size = eval_batch_size
    if checkpoint_interval is not None:
        exp_config.trainer.checkpoint_interval = checkpoint_interval
        exp_config.trainer.summary_interval = checkpoint_interval
        exp_config.trainer.validation_interval = checkpoint_interval

    if learning_rate is not None:
        if exp_config.trainer.optimizer_config.learning_rate.type == 'cosine':
            exp_config.trainer.optimizer_config.learning_rate.cosine.initial_learning_rate = learning_rate
        elif exp_config.trainer.optimizer_config.learning_rate.type == 'constant':
            exp_config.trainer.optimizer_config.learning_rate.constant.learning_rate = learning_rate

    # Check if training already completed
    latest_cp = tf.train.latest_checkpoint(model_dir)
    if os.path.exists(completed_file) and not force:
        print("==============================================================")
        print(f"[INFO] Training is ALREADY COMPLETED for model_dir: {model_dir}")
        print(f"[INFO] Marker file exists: {completed_file}")
        print("==============================================================")

        if latest_cp:
            print(f"[INFO] Exporting saved model from latest checkpoint ({latest_cp}) to {output_dir}...")
            save_options = tf.saved_model.SaveOptions(experimental_custom_gradients=True)
            export_saved_model_lib.export_inference_graph(
                input_type='image_tensor',
                batch_size=1,
                input_image_size=[cfg.HEIGHT, cfg.WIDTH],
                params=exp_config,
                log_model_flops_and_params=True,
                save_options=save_options,
                checkpoint_path=latest_cp,
                export_dir=output_dir)
            print("[INFO] Export complete.")
        return None, {}

    # Set mixed precision policy if configured
    if exp_config.runtime.mixed_precision_dtype == tf.float16:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # Setup distribution strategy
    logical_device_names = [
        logical_device.name for logical_device in tf.config.list_logical_devices()]

    if 'GPU' in ''.join(logical_device_names):
        distribution_strategy = tf.distribute.MirroredStrategy()
        print(f"[INFO] Using MirroredStrategy with GPUs: {logical_device_names}")
    else:
        print('[WARNING] No GPU detected. Falling back to CPU OneDeviceStrategy')
        distribution_strategy = tf.distribute.OneDeviceStrategy(logical_device_names[0])

    print("==============================================================")
    print(f"[INFO] Model Directory:     {model_dir}")
    print(f"[INFO] Output Directory:    {output_dir}")
    print(f"[INFO] Train TFRecord Path: {train_data_dir}")
    print(f"[INFO] Eval TFRecord Path:  {eval_data_dir}")
    print(f"[INFO] Total Target Steps:  {exp_config.trainer.train_steps}")
    if latest_cp:
        print(f"[INFO] Latest Checkpoint Found: {latest_cp} -> RESUMING training.")
    else:
        print(f"[INFO] No existing checkpoint found -> STARTING new training.")
    print("==============================================================")

    with distribution_strategy.scope():
        task = tfm.core.task_factory.get_task(
            exp_config.task, logging_dir=model_dir)

    model, eval_logs = tfm.core.train_lib.run_experiment(
        distribution_strategy=distribution_strategy,
        task=task,
        mode='train_and_eval',
        params=exp_config,
        model_dir=model_dir,
        run_post_eval=True)

    # Check if target steps reached
    latest_cp_after = tf.train.latest_checkpoint(model_dir)
    current_step = 0
    if latest_cp_after:
        try:
            # Checkpoint names end with step index (e.g. ckpt-200000)
            current_step = int(latest_cp_after.split('-')[-1])
        except ValueError:
            pass

    if current_step >= exp_config.trainer.train_steps:
        print(f"\n[INFO] Target training steps reached ({current_step}/{exp_config.trainer.train_steps}).")
        with open(completed_file, "w", encoding="utf-8") as f:
            f.write(f"Completed at step {current_step} / {exp_config.trainer.train_steps}\n")

    # Export inference model
    if latest_cp_after:
        print(f"[INFO] Exporting final inference model to {output_dir}...")
        save_options = tf.saved_model.SaveOptions(experimental_custom_gradients=True)
        export_saved_model_lib.export_inference_graph(
            input_type='image_tensor',
            batch_size=1,
            input_image_size=[cfg.HEIGHT, cfg.WIDTH],
            params=exp_config,
            log_model_flops_and_params=True,
            save_options=save_options,
            checkpoint_path=latest_cp_after,
            export_dir=output_dir)
        print(f"[INFO] SavedModel successfully exported to: {output_dir}")

    return model, eval_logs


if __name__ == "__main__":
    # Register Slurm timeout signal handlers
    signal.signal(signal.SIGUSR1, handle_slurm_signal)
    signal.signal(signal.SIGTERM, handle_slurm_signal)

    parser = argparse.ArgumentParser(description="Resume or start Object Detection model training.")
    parser.add_argument("--model_dir", "-m", type=str,
                        default="/storage/home/deig/nobackup/cv4cdd/data/model_training_logging/experiment_resume",
                        help="Directory to save/load model checkpoints.")
    parser.add_argument("--output_dir", "-o", type=str,
                        default="/storage/home/deig/nobackup/cv4cdd/data/output/exported_model",
                        help="Directory to export the final saved model.")
    parser.add_argument("--train_data_dir", "-t", type=str,
                        default="/storage/home/deig/nobackup/cv4cdd/data/tf_records/train/model_4d-00000-of-00001.tfrecord",
                        help="Path to training TFRecords file.")
    parser.add_argument("--eval_data_dir", "-e", type=str,
                        default="/storage/home/deig/nobackup/cv4cdd/data/tf_records/val/model_4d-00000-of-00001.tfrecord",
                        help="Path to validation TFRecords file.")
    parser.add_argument("--train_steps", type=int, default=None,
                        help="Total number of training steps.")
    parser.add_argument("--train_batch_size", type=int, default=None,
                        help="Global train batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Global evaluation batch size.")
    parser.add_argument("--checkpoint_interval", type=int, default=None,
                        help="Checkpoint save interval in steps.")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Initial learning rate.")
    parser.add_argument("--gpu_devices", type=str, default="",
                        help="CUDA_VISIBLE_DEVICES string (e.g. '0' or '0,1').")
    parser.add_argument("--force", action="store_true",
                        help="Force execution even if TRAINING_COMPLETED exists.")

    args = parser.parse_args()

    if args.gpu_devices:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    run_resume_training(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        train_data_dir=args.train_data_dir,
        eval_data_dir=args.eval_data_dir,
        train_steps=args.train_steps,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        checkpoint_interval=args.checkpoint_interval,
        learning_rate=args.learning_rate,
        force=args.force
    )
