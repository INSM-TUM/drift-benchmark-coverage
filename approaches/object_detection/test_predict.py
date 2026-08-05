import utils.config as cfg
import os
import argparse
import tensorflow as tf

import utils.prediction as pred

image_dir = cfg.TEST_IMAGE_DATA_DIR

model = tf.saved_model.load(cfg.TRAINED_MODEL_PATH)

pred.predict(image_dir=image_dir,
            output_path=cfg.DEFAULT_OUTPUT_DIR,
            model=model,
            encoding_type=cfg.ENCODING_TYPE,
            n_windows=200)