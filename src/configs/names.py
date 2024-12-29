"""Names"""

###############################################################
#                                                             #
#                          ML CONFIG                          #
#                                                             #
###############################################################

### MODELS

MODEL_TYPE = "model_type"
MODEL_UNET = "UNET"  # TO CHANGE
MODEL_WAVEUNET = "WAVEUNET"

ACCURACY = "accuracy"
RECALL = "recall"

### FEATURES

TARGET = "target"
PREDICTION = "prediction"
FEATURES = "features"
COLS_ID = "cols_id"

### PARAMS

LEARNING_RATE = "learning_rate"
NB_EPOCHS = "nb_epochs"
BETAS = "betas"
BATCH_SIZE = "batch_size"

TRAIN_RATIO = "train_ratio"
FEATURE_SELECTION = "feature_selection"
TRAINING_PARAMS = "training_params"
CROSS_VALIDATION = "cross_validation"
FINE_TUNING = "fine_tuning"

# WAVEUNET

NB_CHANNELS_INPUT = "nb_channels_input"
NB_CHANNELS_OUTPUT = "nb_channels_output"
NB_FILTERS = "NB_FILTERS"
N_OUTPUTS = "n_outputs"
KERNEL_SIZE = "kernel_size"
DEPTH = "depth"
