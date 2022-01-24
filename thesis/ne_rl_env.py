"""RL environment for Poker, using an API similar to OpenAI Gym."""
from thesis import pypoker as pypoker
from pypoker import PokerMoveType
from typing import Union

# -------------------------------------------------------------------------------
# Environment API
# -------------------------------------------------------------------------------


class Environment(object):
    """Abstract Environment interface.

    All concrete implementations of an environment should derive from this
    interface and implement the method stubs.
    """

    def reset(self, config):
        """Reset the environment with a new config.

        Signals environment handlers to reset and restart the environment using
        a config dict.

        Args:
          config: dict, specifying the parameters of the environment to be
            generated.

        Returns:
          observation: A dict containing the full observation state.
        """
        raise NotImplementedError("Not implemented in Abstract Base class")

    def step(self, action):
        """Take one step in the game.

        Args:
          action: dict, mapping to an action taken by an agent.

        Returns:
          observation: dict, Containing full observation state.
          reward: float, Reward obtained from taking the action.
          done: bool, Whether the game is done.
          info: dict, Optional debugging information.

        Raises:
          AssertionError: When an illegal action is provided.
        """
        raise NotImplementedError("Not implemented in Abstract Base class")


class PokerEnv(Environment):
    """RL interface to a Poker environment.

    ```python

    environment = ne_rl_env.make()
    config = { 'players': 6 }
    observation = environment.reset(config)
    while not done:
        # Agent takes action
        action =  ...
        # Environment take a step
        observation, reward, done, info = environment.step(action)
    ```
    """

    def __init__(self, config):
        pass

    def reset(self, config=None):
        pass

    def step(self, action: Union[int, PokerMoveType]):
        """Take one step in the game.

        Args:
            action: int, mapping to a legal action taken by this agent. The following
                actions are supported:
                  - 'FOLD': 0
                  - 'CHECK': 1
                  - 'CALL': 2
                  - 'RAISE_3BB': 3
                  - 'RAISE_HALF_POT': 4
                  - 'RAISE_POT': 5
                  - 'RAISE_2POT': 6
                  - 'RAISE_ALL_IN': 7
                  - 'SMALL_BLIND': 8
                  - 'BIG_BLIND': 9

        Returns:
          observation: dict, containing the full observation about the game at the
            current step. *WARNING* This observation contains all the hands of the
            players and should not be passed to the agents.
            An example observation:
            {...
            todo
            }
        """
        # return (observation, reward, done, info
        pass

    def _make_observation_all_players(self):
        pass

    def _extract_dict_from_backend(self):
        pass

    def _build_move(self):
        pass


def make(environment_name="NLHE-Full", num_players=6):
    """Make an environment.

    Args:
      environment_name: str, Name of the environment to instantiate.
      num_players: int, Number of players in this game.

    Returns:
      env: An `Environment` object.

    Raises:
      ValueError: Unknown environment name.
    """

    if environment_name == "NLHE-Full":
        return PokerEnv(
            config={
                "num_players":
                    num_players,
                "observation_type":
                    pypoker.AgentObservationType.STANDARD.value
            })
    # elif environment_name == "NLHE-Small":
    #     return PokerEnv(
    #         config={
    #             "num_players":
    #                 num_players,
    #             "observation_type":
    #                 pypoker.AgentObservationType.STANDARD.value
    #         })
    else:
        raise ValueError("Unknown environment {}".format(environment_name))


# -------------------------------------------------------------------------------
# Poker Agent API
# -------------------------------------------------------------------------------


class Agent(object):
    """Agent interface.

    All concrete implementations of an Agent should derive from this interface
    and implement the method stubs.


    ```python

    class MyAgent(Agent):
      ...

    agents = [MyAgent(config) for _ in range(players)]
    while not done:
      ...
      for agent_id, agent in enumerate(agents):
        action = agent.act(observation)
        if obs.current_player == agent_id:
          assert action is not None
        else
          assert action is None
      ...
    ```
    """

    def __init__(self, config, *args, **kwargs):
        r"""Initialize the agent.

        Args:
          config: dict, With parameters for the game. Config takes the following
            keys and values. todo
              - ...
              - ...
              - players: int, Number of players \in [2,6].
              - seed: int, Random seed.
          *args: Optional arguments
          **kwargs: Optional keyword arguments.

        Raises:
          AgentError: Custom exceptions.
        """
        raise NotImplementedError("Not implemeneted in abstract base class.")

    def reset(self, config):
        r"""Reset the agent with a new config.

        Signals agent to reset and restart using a config dict.

        Args:
          config: dict, With parameters for the game. Config takes the following
            keys and values. todo
              - ...
              - ...
              - players: int, Number of players \in [2,6].
              - seed: int, Random seed.
        """
        raise NotImplementedError("Not implemeneted in abstract base class.")

    def act(self, observation):
        """Act based on an observation.

        Args:
          observation: dict, containing observation from the view of this agent.
            An example: todo
            {'KEY1': VAL1,
             'KEY2': [[{...}...]]],
             'KEYN': VALN}]}

        Returns:
          action: int, mapping to a legal action taken by this agent. The following
                actions are supported:
                  - 'FOLD': 0
                  - 'CHECK': 1
                  - 'CALL': 2
                  - 'RAISE_3BB': 3
                  - 'RAISE_HALF_POT': 4
                  - 'RAISE_POT': 5
                  - 'RAISE_2POT': 6
                  - 'RAISE_ALL_IN': 7
                  - 'SMALL_BLIND': 8
                  - 'BIG_BLIND': 9
        """
        raise NotImplementedError("Not implemented in Abstract Base class")
