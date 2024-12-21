"""Parameters for ML models"""

from src.configs import constants, names


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
    0: {names.MODEL_TYPE: names.MODEL_UNET, "epochs": 40},
    1: {},
    2: {},
    # Add more experiments as needed
}
