"""abstract pipeline"""

import abc


class Pipeline(abc.ABC):
    """abstract pipeline object"""

    def __init__(self, id_experiment: int | None):
        self.id_experiment = id_experiment

    @abc.abstractmethod
    def full_pipeline(self, data_train, data_test):
        """full pipeline object"""
        raise NotImplementedError

    @abc.abstractmethod
    def learning_pipeline(self, data_train, data_test):
        """learning pipeline object"""
        raise NotImplementedError

    @abc.abstractmethod
    def testing_pipeline(self, data_train, data_test):
        """testing pipeline object"""
        raise NotImplementedError
