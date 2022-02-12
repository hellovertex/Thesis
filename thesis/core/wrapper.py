# todo: augment state
# todo: optional, change FOLD replacements to be less aggressive
# todo from action-tuple and observation, construct discrete action (i.e. discretize raise amount)
"""
  # todo: check how obs is normalized to avoid small floats

  # *** Observation Augmentation *** #
  # raise vs bet: raise only in preflop stage, bet after preflop
  actions_per_stage = self._init_player_actions(player_info)

  for stage, actions in episode.actions_total.items():
      for action in actions:
          # noinspection PyTypeChecker
          actions_per_stage[action.player_name][stage].append((action.action_type, action.raise_amount))

  # todo: augment inside env wrapper
  # --- Append last 8 moves per player --- #
  # --- Append all players hands --- #
      """
from PokerEnv.PokerRL.game._.EnvWrapperBuilderBase import EnvWrapperBuilderBase
from PokerEnv.PokerRL.game._.rl_env.base.PokerEnv import PokerEnv
from collections import defaultdict, deque
from typing import Tuple

from thesis.core.encoder import PlayerInfo
from thesis.canonical_vectorizer import CanonicalVectorizer


class Wrapper:

  def __init__(self, env, env_bldr_that_built_me):
    """
    Args:
        env (PokerEnv subclass instance):   The environment instance to be wrapped

        env_bldr_that_built_me:          EnvWrappers should only be created by EnvBuilders. The EnvBuilder
                                            instance passes ""self"" as the value for this argument.
    """
    assert issubclass(type(env), PokerEnv)
    self.env = env
    self.env_bldr = env_bldr_that_built_me

  # _______________________________ directly interact with the env inside the wrapper ________________________________
  def step(self, action):
    """
    Steps the environment from an action of the natural action representation to the environment.

    Returns:
        obs, reward, done, info
    """
    env_obs, rew_for_all_players, done, info = self.env.step(action)
    self._pushback_action(action)
    return env_obs, rew_for_all_players, done, info
    # self._pushback(env_obs)
    # return self._return_obs(env_obs=env_obs, rew_for_all_players=rew_for_all_players, done=done, info=info)

  def step_from_processed_tuple(self, action):
    """
    Steps the environment from a tuple (action, num_chips,).

    Returns:
        obs, reward, done, info
    """
    env_obs, rew_for_all_players, done, info = self.env.step_from_processed_tuple(action)
    return env_obs, rew_for_all_players, done, info

  def step_raise_pot_frac(self, pot_frac):
    """
    Steps the environment from a fractional pot raise instead of an action as usually specified.

    Returns:
        obs, reward, done, info
    """
    env_obs, rew_for_all_players, done, info = self.env.step_raise_pot_frac(pot_frac=pot_frac)
    return env_obs, rew_for_all_players, done, info

  def reset(self, deck_state_dict=None):
    env_obs, rew_for_all_players, done, info = self.env.reset(deck_state_dict=deck_state_dict)
    return env_obs, rew_for_all_players, done, info

  def _pushback_action(self, action):
    raise NotImplementedError


class AugmentObservationWrapper(Wrapper):

  def __init__(self, env, env_bldr_that_built_me,
               table: Tuple[PlayerInfo], player_hands, vectorizer=None):
    super().__init__(env=env, env_bldr_that_built_me=env_bldr_that_built_me)
    self._table = table
    # augmentation content
    self._actions_per_stage = None
    self._player_hands = player_hands
    # vectorizes augmented observation
    # todo add kwargs from table to initialization of vectorizer
    self._vectorizer = CanonicalVectorizer() if vectorizer is None else vectorizer

  # noinspection PyTypeChecker
  def _init_player_actions(self):
    # todo initialize from Positions instead of player names
    player_actions = {}
    for p_info in self._table:
      # create default dictionary for current player for each stage
      # default dictionary stores only the last two actions per stage per player
      player_actions[p_info.player_name] = defaultdict(
        lambda: deque(maxlen=2), keys=['preflop', 'flop', 'turn', 'river'])
    self._actions_per_stage = player_actions
    return player_actions

  def pushback_action(self, action_formatted):
    pass

  @property
  def current_player(self):
    return self.env.current_player


class AugmentedEnvBuilder(EnvWrapperBuilderBase):
  WRAPPER_CLS = AugmentObservationWrapper

  def __init__(self, env_cls, env_args):
    super().__init__(env_cls=env_cls, env_args=env_args)

  def _get_num_public_observation_features(self):
    # todo call encoder_that_built_me._get_num_public_observation_features
    return super()._get_num_public_observation_features()

  def _get_num_private_observation_features(self):
    return super()._get_num_private_observation_features()

  def _get_obs_parts_idxs(self):
    return super()._get_obs_parts_idxs()
