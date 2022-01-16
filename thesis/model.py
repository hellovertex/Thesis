""" This module will contain top level definition of Poker Model"""
from rl.agents import DQNAgent
import pandas as pd


class PokerModel:
    """ class variables to keep track of rmse per estimator """
    rmse = []
    estimators = []

    def __init__(self, params: dict):
        """
        :param params: dictionary containing hyperparams of the model
        """
        self._params = params
        self._model = DQNAgent(params)

    @classmethod
    def new_instance(cls, params: dict):
        """
        :param params: Used to create new class instance model
        """
        return cls(params)

    @property
    def model(self):
        """ Getter for the wrapped model"""
        return self._model

    @property
    def params(self):
        """ Getter for params of wrapped model """
        return self._params

    def mlflow_run(self, data_frame: pd.DataFrame, run_name: str = "Baseline"):
        """ Training and MLFlow logging will happen here """
