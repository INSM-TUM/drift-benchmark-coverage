#### GENERAL CONFIG #####
DEBUG = True
ANNOTATIONS_ONLY = False
AUTOMATE_TFR_SCRIPT = False
VDD_PREPROCESSING = False
KEEP_AXIS = False
WINDOWS_SYSTEM = False
MINE_CONSTRAINTS = False
CONSTRAINTS_DIR = ""

if VDD_PREPROCESSING:
    ENCODING_TYPE = "vdd"
else:
    ENCODING_TYPE = "winsim"


##### DATA CONFIG #####
N_CORES_WINSIM = 100
N_WINDOWS = 200
DEFAULT_DATA_DIR = "/work/alexkrau/projects/scdd/data/input_4d/test" # "Specify default data output directory". Usable also with "val" and "test"
DEFAULT_LOG_DIR =  "/work/alexkrau/projects/scdd/data/input_4d/test" # "Specify event log directory". Usable also with "val" and "test"
TFR_RECORDS_DIR =  "/work/alexkrau/projects/scdd/data/tf_records/test" # "Specify directory where to save TFR files here". Usable also with "val" and "test"
# when using paths for the folder "test", switch the parameter "AUTOMATE_TFR_SCRIPT" to "False"
OUTPUT_PREFIX = "model_4d"

ANNOTATION_TYPE = "only_complex"
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
TRAIN_EXAMPLES = 20000
EVAL_EXAMPLES = 1250
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
TRAIN_DATA_DIR =       "/work/alexkrau/projects/scdd/data/tf_records/train/model_4d-00000-of-00001.tfrecord" # "Specify path to TFR training dataset here"
EVAL_DATA_DIR =        "/work/alexkrau/projects/scdd/data/tf_records/val/model_4d-00000-of-00001.tfrecord" #"Specify path to TFR validation dataset here"
DEFAULT_OUTPUT_DIR =   "/work/alexkrau/projects/scdd/data/output" #"Specify directory where to save output here"

###################################
##### Evaluation ##################
###################################
EVAL_THRESHOLD = 0.5
TRAINED_MODEL_PATH = "/work/alexkrau/projects/scdd/data/output/20240302-004101_winsim_sgd" # "Specify path to TFR training dataset here"

# test winsim figures
TEST_IMAGE_DATA_DIR =  "C:\\Users\\alkraus\\ResearchGit\\cv4cdd_main\\data\\input_4d\\test\\winsim\\experiment_20240919-173212" #"Specify directory where evaluation images are saved here"
DRIFT_INFO_INITIAL =   "C:\\Users\\alkraus\\ResearchGit\\cv4cdd_main\\data\\input_4d\\test" #Specify directory where the drift info files is stored"
#TEST_IMAGE_DATA_DIR =  "/work/alexkrau/projects/scdd/data/input_4d/test/" #"Specify directory where evaluation images are saved here"
#DRIFT_INFO_INITIAL =   "/work/alexkrau/projects/scdd/data/input_4d/test/" #"Specify directory where the drift info files is stored"

##### EVALUATION CONFIG #####
RELATIVE_LAG = [0.01, 0.025, 0.05]
EVAL_MODE = "general"
#PRODRIFT_DIR = "/.../scdd/ProDrift2.5"
VDD_DIR = ""
