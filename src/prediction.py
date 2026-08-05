import os
import json
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

import src.preprocessing as pp
import src.evaluation as eval
import src.utilities as utils

from tqdm import tqdm
from typing import List


def prediction_pipeline(log_dir: str, encoding_type: str, output_path: str,
                        cp_all=None) -> str:
    """Entry script for prediction preprocessing pipeline.

    Args:
        log_dir (str): Event log directory
        encoding_type (str): Preprocessing encoding method
        output_path (str): Output directory
        cp_all (bool, optional): VDD measure. Defaults to None.

    Returns:
        str: Image directory
    """
    image_dir = os.path.join(output_path, "winsim_images")

    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)

    winsim_pipeline(log_dir, output_path=image_dir)
    return image_dir


def winsim_pipeline(log_dir: str, output_path: str):
    """WINSIM pipeline for encoding logs for prediction. 

    Args:
        log_dir (str): Log directory
        output_path (str): Output path
    """
    # get all paths and file names of event logs
    log_files = utils.get_event_log_paths(log_dir)

    window_info = {}

    # iterate through log files
    for name, path in tqdm(log_files.items(), desc="Preprocessing Event Logs",
                           unit="Event Log"):
        real_name = name.split(".")[0]
        # load event log
        event_log = utils.import_event_log(path=path, name=name)

        # if the log contains incomplete traces, the log is filtered
        filtered_log = utils.filter_complete_events(event_log)

        import src.config as cfg
        if cfg.BASE_REPRESENTATION in ["dfg", "dfg_io"]:
            windowed_matrices, _, window_information, _ = \
                pp.log_to_windowed_dfg_count(filtered_log, add_io=(cfg.BASE_REPRESENTATION == "dfg_io"))
        else:
            windowed_matrices, _, window_information, _ = \
                pp.log_to_windowed_arm(filtered_log)

        window_info[real_name] = window_information

        # get similarity matrix
        sim_matrix = pp.similarity_calculation(windowed_matrices)

        # save matrix as image
        utils.matrix_to_img(matrix=sim_matrix,
                            number=real_name,
                            exp_path=output_path,
                            mode="color")

    window_info_path = os.path.join(output_path, "window_info.json")
    with open(window_info_path, "w", encoding='utf-8') as file:
        json.dump(window_info, file)


def save_pred_results(results: dict, output_path: str):
    """Saves prediction results to output path.

    Args:
        results (dict): Evaluation measures
        output_path (str): Evaluation directory
    
    Returns:
        pd.DataFrame: DataFrame, containing predictions
    """
    results_df = pd.DataFrame.from_dict(results, orient="index")
    save_path = os.path.join(output_path, "prediction_results.csv")
    results_df.to_csv(save_path, sep=",")


def get_changepoints_trace_idx_winsim(bboxes: list, y_pred: list,
                                      window_info: dict) -> List[tuple]:
    """Get changepoints as trace indices for WINSIM encoding. 

    Args:
        bboxes (list): List of bboxes
        y_pred (list): List of predictions
        window_info (dict): Window info

    Returns:
        List[tuple]: List of changepoints as trace indices
    """
    change_points = []
    if len(bboxes) == 0:
        return change_points
    else:
        for i, bbox in enumerate(bboxes):
            if y_pred[i] == "sudden":
                # changepoint is equal to the date of the first trace in the middle
                # window of bbox
                change_point = eval.get_sudden_changepoint_winsim(
                    round(bbox[0]))
                change_point_trace_id = (window_info[str(change_point)][0],
                                         window_info[str(change_point)][0])
            else:
                # change start and end is equal to the date of the first trace in window
                # set window id of change start to at least 1 for edge cases
                # set window id of change end to maximum 200 for edge cases
                change_start = (1 if round(bbox[0]) < 1 else round(bbox[0]))
                change_end = (200 if round(bbox[2]) > 200 else round(bbox[2]))
                change_point_trace_id = (window_info[str(change_start)][0],
                                         window_info[str(change_end)][0])
            change_points.append(change_point_trace_id)
        return change_points


def visualize_prediction(path: str, image: np.ndarray, image_name: str,
                         bbox_pred: np.ndarray, y_pred: np.ndarray, score: np.ndarray,
                         encoding: str):
    """Visualize predicted bounding boxes for an image.

    Args:
        path (str): Output path
        image (np.ndarray): Image as numpy array
        image_name (str): Image name
        bbox_pred (list): Predicted bounding box
        y_pred (list): Predicted classes
        score (list): Confidence score
        encoding (str): Encoding type
    """
    category_index, _ = utils.get_ex_decoder()

    plt.figure(figsize=(10, 10))
    
    image = image[0].numpy()

    utils.visualize_boxes_and_labels(image=image,
                                     bboxes=bbox_pred,
                                     labels=y_pred,
                                     score=score,
                                     category_index=category_index,
                                     is_groundtruth=False,
                                     encoding=encoding)
    plt.imshow(image)
    plt.axis('off')

    plt.savefig(os.path.join(path, f"{image_name}.png"),
                bbox_inches="tight")
    plt.close()


def predict(image_dir: str, output_path: str, model: tf.keras.Model,
            encoding_type: str):
    """Main prediction script.

    Args:
        image_dir (str): Directory containing images for prediction
        output_path (str): Output path
        model (tf.keras.Model): Trained model
        encoding_type (str): Name of encoding method

    Raises:
        ValueError: Raises ValueError if no valid encoding type is specified
    """
    input_image_size = (256, 256)
    targetsize = 256
    threshold = 0.5
    model_fn = model.signatures['serving_default']
    pred_results = {}
    
    if os.path.isfile(os.path.join(image_dir,"log_matching.csv")):
            log_matching = eval.get_log_matching(image_dir)
    else:
        log_matching = None

    window_info = eval.get_window_info(image_dir)

    category_index, _ = utils.get_ex_decoder()

    images = eval.get_image_paths(image_dir)

    import src.config as cfg
    n_windows = cfg.N_WINDOWS

    for image_name, image_path in tqdm(images.items(),
                                       desc="Detecting Concept Drift", unit="images"):

        path = os.path.join(image_path, image_name)
        image_name = image_name.split(".")[0]
        image = utils.load_image(path)
        image = utils.build_inputs_for_object_detection(
            image, input_image_size)
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(image, dtype=tf.uint8)
        result = model_fn(image)

        scores = result['detection_scores'][0].numpy()
        confidence_scores = scores[scores > threshold]

        bbox_pred = result['detection_boxes'][0].numpy()
        bbox_pred = bbox_pred[scores > threshold]

        y_pred = result['detection_classes'][0].numpy().astype(int)
        y_pred = y_pred[scores > threshold]

        y_pred_category = eval.get_predicted_classes(y_pred, category_index)

        visualize_prediction(path=output_path,
                             image=image,
                             image_name=image_name,
                             bbox_pred=bbox_pred,
                             y_pred=y_pred,
                             score=confidence_scores,
                             encoding=encoding_type)

        bbox_pred = bbox_pred / targetsize \
            * n_windows
        if log_matching is not None:
            log_name = log_matching.loc[log_matching["image_id"] == 
                                        int(image_name), "log_name"].iloc[0]
            log_window_info = window_info[log_name]
        else:
            log_window_info = window_info[image_name]
        pred_change_points = get_changepoints_trace_idx_winsim(
            bbox_pred, y_pred_category, log_window_info)
        pred_results[image_name] = \
            {"Detected Changepoints": pred_change_points,
                "Detected Drift Types": y_pred_category,
                "Prediction Confidence": np.round(confidence_scores, decimals=4)}

    save_pred_results(pred_results, output_path)
