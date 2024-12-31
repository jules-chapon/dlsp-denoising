"""Parameters for ML models"""

from src.configs import names


###############################################################
#                                                             #
#                           CONSTANTS                         #
#                                                             #
###############################################################

NB_OPTUNA_TRIALS = 3

###############################################################
#                                                             #
#                     EXPERIMENTS CONFIGS                     #
#                                                             #
###############################################################

# 0: Gaetano
# 1: Coco
# 2: Jules

EXPERIMENTS_CONFIGS = {
    0: {
        names.MODEL_TYPE: names.MODEL_UNET,
        names.NB_EPOCHS: 150,
        names.LEARNING_RATE: 0.01,
    },
    1: {},
    200: {
        names.MODEL_TYPE: names.MODEL_WAVEUNET,
        names.NB_EPOCHS: 100,
        names.LEARNING_RATE: 0.0001,
        names.BETAS: [0.9, 0.999],
        names.BATCH_SIZE: 16,
        names.NB_CHANNELS_INPUT: 1,
        names.NB_CHANNELS_OUTPUT: 1,
        names.NB_FILTERS: 24,
        names.DEPTH: 4,
    },
    # Add more experiments as needed
}
