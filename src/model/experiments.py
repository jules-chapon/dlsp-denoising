"""Functions to define different experiments"""

from src.configs import ml_config, names

from src.model.pipeline import Pipeline

from src.model.pipeline_unet import PipelineUnet


def init_pipeline_from_config(id_experiment: int) -> Pipeline | None:
    """
    Initialize a model for a given experiment.

    Args:
        id_experiment (int): ID of the experiment.

    Returns:
        _LGBMModel | None: Model with the parameters of the given experiment.
    """
    config = ml_config.EXPERIMENTS_CONFIGS[id_experiment]
    if config[names.MODEL_TYPE] == names.MODEL_UNET:
        return PipelineUnet(id_experiment=id_experiment)
    else:
        return None
