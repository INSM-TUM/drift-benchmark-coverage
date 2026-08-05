#### GENERAL CONFIG #####
DEBUG = True
ANNOTATIONS_ONLY = False
AUTOMATE_TFR_SCRIPT = False #True
VDD_PREPROCESSING = False
KEEP_AXIS = False
WINDOWS_SYSTEM = False
MINE_CONSTRAINTS = False
CONSTRAINTS_DIR = ""

if VDD_PREPROCESSING:
    ENCODING_TYPE = "vdd"
else:
    ENCODING_TYPE = "winsim"

BASE_REPRESENTATION = "dfg" # Options: 'dfg', 'dfg_io', 'arm'

##### DATA CONFIG #####
N_CORES_WINSIM = 170 #50
N_WINDOWS = 200
DEFAULT_DATA_DIR = "/work/alexkrau/projects/scdd/data/input_4d/test" # "Specify default data output directory". Usable also with "val" and "test"
DEFAULT_LOG_DIR =  "/work/alexkrau/projects/scdd/data/input_4d/test" # "Specify event log directory". Usable also with "val" and "test"
TFR_RECORDS_DIR =  "/work/alexkrau/projects/scdd/data/tf_records/test" # "Specify directory where to save TFR files here". Usable also with "val" and "test"

# when using paths for the folder "test", switch the parameter "AUTOMATE_TFR_SCRIPT" to "False"
OUTPUT_PREFIX = "model_4d_cd"

ANNOTATION_TYPE = "complex_detailed"
# Three options are possible
# "default" - annotations are created for each drift based on its start and end change points
# "only_complex" - annotations are only created for the start and end change points of complex drifts
# "complex_detailed" - annotations are created for all drifts, including sudden and gradual changes as part of a complex drift

#####
TENSORFLOW_MODELS_DIR = "/work/alexkrau/projects/scdd/models" #"Specify TensorFlow model garden directory"
MINERFUL_SCRIPTS_DIR = "/work/alexkrau/projects/scdd/data/MINERful" #"Specify MINERful directory"
DRIFT_TYPES = ["sudden", "gradual", "incremental", "recurring"]
DISTANCE_MEASURE = "cos" # can be one of ["fro","nuc","inf","l2","cos","earth"]
COLOR = "color"
RESIZE_SUDDEN_BBOX = True
RESIZE_VALUE = 5

##### VDD CONFIG #####
SUB_L = 100
SLI_BY = 50
CP_ALL = True


##### MODEL CONFIG #####
FACTOR = 500
TRAIN_EXAMPLES = 40000
EVAL_EXAMPLES = 2500
TRAIN_BATCH_SIZE = 64
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

# Possible Models:
# retinanet_resnetfpn_coco, retinanet_spinenet_coco
MODEL_SELECTION = "retinanet_spinenet_coco"
# ID can be 143 or 190
SPINENET_ID = "143"


###################################
##### MODEL TRAINING ##############
###################################
# stores checkpints for training and validation
MODEL_PATH =           "/work/alexkrau/projects/scdd/data/model_training_logging" #  "Specify directory where to log model training here"

# DEFINE TRAINING-RELEVANT LINKS
# train 4d model with annotation: only_complex
#TRAIN_DATA_DIR =       "/work/alexkrau/projects/scdd/data/tf_records/train/model_4d_only_complex-00000-of-00001.tfrecord" # "Specify path to TFR training dataset here"
#EVAL_DATA_DIR =        "/work/alexkrau/projects/scdd/data/tf_records/val/model_4d_only_complex-00000-of-00001.tfrecord" #"Specify path to TFR validation dataset here"
#DEFAULT_OUTPUT_DIR =   "/work/alexkrau/projects/scdd/data/output" #"Specify directory where to save output here"
# train 4d model with annotation: complex_detailed
TRAIN_DATA_DIR =       "/work/alexkrau/projects/scdd/data/tf_records/train/model_4d-00000-of-00001.tfrecord" # "Specify path to TFR training dataset here"
EVAL_DATA_DIR =        "/work/alexkrau/projects/scdd/data/tf_records/val/model_4d-00000-of-00001.tfrecord" #"Specify path to TFR validation dataset here"
DEFAULT_OUTPUT_DIR =   "/work/alexkrau/projects/scdd/data/output" #"Specify directory where to save output here"

###################################
##### Evaluation ##################
###################################
EVAL_THRESHOLD = 0.5

# Specify path to folder with the trained model
#TRAINED_MODEL_PATH = "/work/alexkrau/projects/scdd/data/output/20240919-223200_winsim_sgd_model_complex_only_v1" # only complex drifts (model v1)
#TRAINED_MODEL_PATH = "/work/alexkrau/projects/scdd/data/output/20240925-102308_winsim_sgd_model_complex_only_v2" # only complex drifts (model v2)
#TRAINED_MODEL_PATH = "/work/alexkrau/projects/scdd/data/output/20241002-112221_winsim_sgd_model_complex_only_v3" # only complex drifts (model v3)

#TRAINED_MODEL_PATH = "/work/alexkrau/projects/scdd/data/output/20240922-233643_winsim_sgd_model_4d_v1" # all four drift types (model v1)
#TRAINED_MODEL_PATH = "/work/alexkrau/projects/scdd/data/output/20240922-234439_winsim_sgd_model_4d_v2" # all four drift types (model v2)
#TRAINED_MODEL_PATH = "/work/alexkrau/projects/scdd/data/output/20241010-220931_winsim_sgd_model_4d_v3" # all four drift types (model v3) new
#TRAINED_MODEL_PATH = "/work/alexkrau/projects/scdd/data/output/20241010-221910_winsim_sgd_model_4d_v4" # all four drift types (model v4) new
#TRAINED_MODEL_PATH = "/work/alexkrau/projects/scdd/data/output/20241012-154327_winsim_sgd_model_4d_v5" # all four drift types (model v5) new
#TRAINED_MODEL_PATH = "/work/alexkrau/projects/scdd/data/output/20241017-172047_winsim_sgd"             # all four drift types (model v6) new
TRAINED_MODEL_PATH = "/home/deig/bathesis/cv4cdd/data/output/20240922-233643_winsim_sgd_model_4d_v1"              # all four drift types (model v7) new


#TRAINED_MODEL_PATH = "/work/alexkrau/projects/scdd/data/output/20240928-202913_winsim_sgd_model_4d_v3" # all four drift types (model v3) - NA
#TRAINED_MODEL_PATH = "/work/alexkrau/projects/scdd/data/output/20240928-205023_winsim_sgd_model_4d_v4" # all four drift types (model v4) - NA
#TRAINED_MODEL_PATH = "/work/alexkrau/projects/scdd/data/output/20241002-111056_winsim_sgd_model_4d_v5" # all four drift types (model v5) - NA


# test winsim figures
TEST_IMAGE_DATA_DIR =  "/work/alexkrau/projects/scdd/data/input_cdlg_4d/test/winsim/experiment_20240919-173212/" # w200"Specify directory where evaluation images are saved"
#TEST_IMAGE_DATA_DIR =  "/work/alexkrau/projects/scdd/data/input_4d/test/winsim/experiment_20241011-104901_w250/" #"Specify directory where evaluation images are saved"
#TEST_IMAGE_DATA_DIR =  "/work/alexkrau/projects/scdd/data/input_4d/test/winsim/experiment_20241011-104752_w150/" #"Specify directory where evaluation images are saved"
#TEST_IMAGE_DATA_DIR =  "/work/alexkrau/projects/scdd/data/input_4d/test/winsim/experiment_20241011-105954_w300/" #"Specify directory where evaluation images are saved"
#TEST_IMAGE_DATA_DIR =  "/work/alexkrau/projects/scdd/data/input_4d/test/winsim/experiment_20241011-110037_w100/" #"Specify directory where evaluation images are saved"
#TEST_IMAGE_DATA_DIR =  "/work/alexkrau/projects/scdd/data/input_4d/test/winsim/experiment_20241011-111000_w125/" #"Specify directory where evaluation images are saved"
#TEST_IMAGE_DATA_DIR =  "/work/alexkrau/projects/scdd/data/input_4d/test/winsim/experiment_20241011-111017_w175/" #"Specify directory where evaluation images are saved"
#TEST_IMAGE_DATA_DIR =  "/work/alexkrau/projects/scdd/data/input_4d/test/winsim/experiment_20241011-111549_w225/" #"Specify directory where evaluation images are saved"
#TEST_IMAGE_DATA_DIR =  "/work/alexkrau/projects/scdd/data/input_4d/test/winsim/experiment_20241011-111601_w275/" #"Specify directory where evaluation images are savede"

DRIFT_INFO_INITIAL =   "/work/alexkrau/projects/scdd/data/input_cdlg_4d/test/" #"Specify directory where the drift info files is stored"

##### EVALUATION CONFIG #####
RELATIVE_LAG = [0.05, 0.01, 0.025]
EVAL_MODE = "general"
#PRODRIFT_DIR = "/.../scdd/ProDrift2.5"
VDD_DIR = ""
