# pylint: disable=unused-variable
""" This module will contain top level definition of Poker Model"""
import mlflow
from rl.agents import DQNAgent
import pandas as pd
# import neuron_poker.gym_env.env as env


class PokerModel:
    """ class variables to keep track of rmse per estimator """
    losses = []
    networks = []

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
        with mlflow.start_run(run_name=run_name) as run:
            # get mlflow run metadata
            run_id = run.info.run_id
            experiment_id = run.info.experiment_id

            # create train and test data
            x = data_frame  # pylint: disable=invalid-name

            # preprocessing
            x_train = x_test = []

            # train model
            train_fn = self._model.fit
            y_pred = []  # train_fn(x_test)

            # log model
            mlflow.keras.log_model(self.model, "PokerDQNModel")

            # log params
            mlflow.log_params(self.params)

            # compute eval metrics
            loss = 0
            acc = 0

            # log metrics
            mlflow.log_metric("loss", loss)
            mlflow.log_metric("acc", acc)

            # track rmse and estimators
            self.losses.append(loss)
            self.networks.append(self.params)

            # save artifacts
            plot = "someplot"
            tmp_dir = "."
            # # fig.save_fig(tmp_dir)
            # # mlflow.log_artifact(tmp_dir, f"Current_Model_{0}"

            return experiment_id, run_id

    # def action(self, reward, current_player, legal_actions, observation):
    #     """Stores observations from last transition and chooses a new action.
    #
    #     Notifies the agent of the outcome of the latest transition and stores it
    #       in the replay memory, selects a new action and applies a training step.
    #
    #     Args:
    #       reward: float, the reward received from its action.
    #       current_player: int, the player whose turn it is.
    #       legal_actions: `np.array`, actions which the player can currently take.
    #       observation: `np.array`, the most recent observation.
    #
    #     Returns:
    #       A legal, int-valued action.
    #     """
    #     self._train_step()
    #
    #     self.action = self._select_action(observation, legal_actions)
    #     self._record_transition(current_player, reward, observation, legal_actions,
    #                             self.action)
    #     return self.action