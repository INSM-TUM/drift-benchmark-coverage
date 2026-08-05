import os

#### GENERAL CONFIG #####
DEBUG = True
ANNOTATIONS_ONLY = False
AUTOMATE_TFR_SCRIPT = False
RESUME = False

ENCODING_TYPE = "winsim"
BASE_REPRESENTATION = "arm" # can be "arm", "dfg", or "dfg_io"

##### DATA CONFIG #####
N_CORES_WINSIM = 64
N_WINDOWS = 200
OUTPUT_PREFIX = "model_4d_cd"

# Resolve directories relative to the workspace root for portability
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Not using DEFAULT_DATA_DIR, DEFAULT_LOG_DIR, TFR_RECORDS_DIR globally here 
# since they are dynamically injected at runtime in preprocessing.py

ANNOTATION_TYPE = "complex_detailed"
# Three options are possible
# "default" - annotations are created for each drift based on its start and end change points
# "only_complex" - annotations are only created for the start and end change points of complex drifts
# "complex_detailed" - annotations are created for all drifts, including sudden and gradual changes as part of a complex drift

TENSORFLOW_MODELS_DIR = os.path.join(ROOT_DIR, "models")
DRIFT_TYPES = ["sudden", "gradual", "incremental", "recurring"]
DISTANCE_MEASURE = "cos" # can be one of ["fro","nuc","inf","l2","cos","earth"]
COLOR = "color"
RESIZE_SUDDEN_BBOX = True
RESIZE_VALUE = 5


##### MODEL CONFIG #####
FACTOR = 500
TRAIN_EXAMPLES = 40000
EVAL_EXAMPLES = 2500
TRAIN_BATCH_SIZE = 128
EVAL_BATCH_SIZE = 32
STEPS_PER_LOOP = TRAIN_EXAMPLES // TRAIN_BATCH_SIZE
TRAIN_STEPS = FACTOR * STEPS_PER_LOOP
VAL_STEPS = EVAL_EXAMPLES // EVAL_BATCH_SIZE
SUMMARY_INTERVAL = STEPS_PER_LOOP
CP_INTERVAL = STEPS_PER_LOOP
VAL_INTERVAL = STEPS_PER_LOOP

# must be equally sized!
IMAGE_SIZE = (256, 256)
TARGETSIZE = 256
N_CLASSES = len(DRIFT_TYPES)
SCALE_MAX = 2.0
SCALE_MIN = 0.1

WIDTH, HEIGHT  = IMAGE_SIZE 
LR_DECAY = True
LR_INITIAL = 1e-3
LR_WARMUP = 2.5e-4
LR_WARMUP_STEPS = 0.1 * TRAIN_STEPS

BEST_CP_METRIC = "AP"
BEST_CP_METRIC_COMP = "higher"

OPTIMIZER_TYPE = "sgd"
LR_TYPE = "cosine" #"stepwise"

SGD_MOMENTUM = 0.9
SGD_CLIPNORM = 10.0

ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999

STEPWISE_BOUNDARIES = [0.95 * TRAIN_STEPS,
                       0.98 * TRAIN_STEPS]
STEPWISE_VALUES = [0.32 * TRAIN_BATCH_SIZE / 256.0,
                   0.032 * TRAIN_BATCH_SIZE / 256.0,
                   0.0032 * TRAIN_BATCH_SIZE / 256.0]

# Possible Models: retinanet_resnetfpn_coco, retinanet_spinenet_coco
MODEL_SELECTION = "retinanet_spinenet_coco"
SPINENET_ID = "143"  # ID can be 143 or 190


###################################
##### MODEL TRAINING ##############
###################################
# stores checkpoints for training and validation
MODEL_PATH = os.path.join(ROOT_DIR, "data/model_training_logging")

# DEFINE TRAINING-RELEVANT LINKS
TRAIN_DATA_DIR = os.path.join(ROOT_DIR, "data/tf_records/train/model_4d_cd-00000-of-00001.tfrecord")
EVAL_DATA_DIR = os.path.join(ROOT_DIR, "data/tf_records/val/model_4d_cd-00000-of-00001.tfrecord")
DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "data/output")


###################################
##### Evaluation ##################
###################################
EVAL_THRESHOLD = 0.5

# Specify path to folder with the trained model
TRAINED_MODEL_PATHS = {
    "arm": os.path.join(ROOT_DIR, "outputs/exp_arm_latest/exported_model"),
    "dfg": os.path.join(ROOT_DIR, "outputs/exp_dfg_latest/exported_model"),
    "dfg_io": os.path.join(ROOT_DIR, "outputs/exp_dfg_latest/exported_model"),
}
TRAINED_MODEL_PATH = TRAINED_MODEL_PATHS.get(BASE_REPRESENTATION)

# test winsim figures
TEST_IMAGE_DATA_DIRS = {
    "arm": os.path.join(ROOT_DIR, "outputs/exp_arm_latest/preprocessed_data/test/winsim"),
    "dfg": os.path.join(ROOT_DIR, "outputs/exp_dfg_latest/preprocessed_data/test/winsim"),
    "dfg_io": os.path.join(ROOT_DIR, "outputs/exp_dfg_io_latest/preprocessed_data/test/winsim"),
}
TEST_IMAGE_DATA_DIR = TEST_IMAGE_DATA_DIRS.get(BASE_REPRESENTATION)
DRIFT_INFO_INITIAL = TEST_IMAGE_DATA_DIR


##### EVALUATION CONFIG #####
RELATIVE_LAG = [0.05, 0.01, 0.025]
EVAL_MODE = "general"

###################################
##### Evaluation Data Paths #######
###################################

###################################
##### Input Dataset Paths #########
###################################
CDLG_TRAIN_DIR = os.path.join(ROOT_DIR, "data/input_cdlg/train")
CDLG_VAL_DIR = os.path.join(ROOT_DIR, "data/input_cdlg/val")
CDLG_TEST_DIR = os.path.join(ROOT_DIR, "data/input_cdlg/test")

# CDLG Evaluation
CDLG_TEST_DRIFT_INFO = os.path.join(ROOT_DIR, "data/input_cdlg/test/drift_info.csv")
CDLG_TEST_OUTPUT_DIRS = {
    "arm": os.path.join(ROOT_DIR, "outputs/exp_arm_latest/preprocessed_data/test"),
    "dfg": os.path.join(ROOT_DIR, "outputs/exp_dfg_latest/preprocessed_data/test"),
    "dfg_io": os.path.join(ROOT_DIR, "outputs/exp_dfg_io_latest/preprocessed_data/test"),
}
CDLG_TEST_OUTPUT_DIR = CDLG_TEST_OUTPUT_DIRS.get(BASE_REPRESENTATION)

# CDrift Evaluation
CDRIFT_LOG_DIR = os.path.join(ROOT_DIR, "data/input_cdrift/inscope")
CDRIFT_GT_CSV = os.path.join(ROOT_DIR, "data/input_cdrift/log_size_info.csv")

CDRIFT_OUTPUT_DIRS = {
    "arm": os.path.join(ROOT_DIR, "outputs/exp_arm_latest/cdrift_eval/preprocessed"),
    "dfg": os.path.join(ROOT_DIR, "outputs/exp_dfg_latest/cdrift_eval/preprocessed"),
    "dfg_io": os.path.join(ROOT_DIR, "outputs/exp_dfg_io_latest/cdrift_eval/preprocessed"),
}
CDRIFT_OUTPUT_DIR = CDRIFT_OUTPUT_DIRS.get(BASE_REPRESENTATION)

# Custom Evaluation
CUSTOM_EVAL_ROOT = os.path.join(ROOT_DIR, "data/custom_eval")
CUSTOM_EVAL_OUTPUT_DIRS = {
    "arm": os.path.join(ROOT_DIR, "outputs/exp_arm_latest/custom_evaluation"),
    "dfg": os.path.join(ROOT_DIR, "outputs/exp_dfg_latest/custom_evaluation"),
    "dfg_io": os.path.join(ROOT_DIR, "outputs/exp_dfg_io_latest/custom_evaluation"),
}
CUSTOM_EVAL_OUTPUT_DIR = CUSTOM_EVAL_OUTPUT_DIRS.get(BASE_REPRESENTATION)

# Evaluation Preprocessing Behavior
AUTO_RUN_PREPROCESSING = True

# Default experiment name
EXPERIMENT_NAMES = {
    "arm": "exp_arm",
    "dfg": "exp_dfg",
    "dfg_io": "exp_dfg_io",
}
experiment_name = EXPERIMENT_NAMES.get(BASE_REPRESENTATION, "cv4cdd_default")



# Function to recalculate derived configurations
def recalculate_derived_configs(base_rep=None):
    global BASE_REPRESENTATION
    global WIDTH, HEIGHT, STEPS_PER_LOOP, TRAIN_STEPS, VAL_STEPS, SUMMARY_INTERVAL, CP_INTERVAL, VAL_INTERVAL, LR_WARMUP_STEPS, STEPWISE_BOUNDARIES, STEPWISE_VALUES, N_CLASSES
    global experiment_name, TRAINED_MODEL_PATH, TEST_IMAGE_DATA_DIR, DRIFT_INFO_INITIAL, CDRIFT_OUTPUT_DIR, CDLG_TEST_OUTPUT_DIR, CUSTOM_EVAL_OUTPUT_DIR

    if base_rep:
        BASE_REPRESENTATION = base_rep

    experiment_name = EXPERIMENT_NAMES.get(BASE_REPRESENTATION, "cv4cdd_default")
    TRAINED_MODEL_PATH = TRAINED_MODEL_PATHS.get(BASE_REPRESENTATION)
    TEST_IMAGE_DATA_DIR = TEST_IMAGE_DATA_DIRS.get(BASE_REPRESENTATION)
    DRIFT_INFO_INITIAL = TEST_IMAGE_DATA_DIR
    CDRIFT_OUTPUT_DIR = CDRIFT_OUTPUT_DIRS.get(BASE_REPRESENTATION)
    CDLG_TEST_OUTPUT_DIR = CDLG_TEST_OUTPUT_DIRS.get(BASE_REPRESENTATION)
    CUSTOM_EVAL_OUTPUT_DIR = CUSTOM_EVAL_OUTPUT_DIRS.get(BASE_REPRESENTATION)

    WIDTH, HEIGHT = IMAGE_SIZE
    N_CLASSES = len(DRIFT_TYPES)
    STEPS_PER_LOOP = TRAIN_EXAMPLES // TRAIN_BATCH_SIZE
    TRAIN_STEPS = FACTOR * STEPS_PER_LOOP
    VAL_STEPS = EVAL_EXAMPLES // EVAL_BATCH_SIZE
    SUMMARY_INTERVAL = STEPS_PER_LOOP
    CP_INTERVAL = STEPS_PER_LOOP
    VAL_INTERVAL = STEPS_PER_LOOP
    LR_WARMUP_STEPS = int(0.1 * TRAIN_STEPS)
    STEPWISE_BOUNDARIES = [int(0.95 * TRAIN_STEPS), int(0.98 * TRAIN_STEPS)]
    STEPWISE_VALUES = [
        0.32 * TRAIN_BATCH_SIZE / 256.0,
        0.032 * TRAIN_BATCH_SIZE / 256.0,
        0.0032 * TRAIN_BATCH_SIZE / 256.0
    ]

# Run recalculation
recalculate_derived_configs()
