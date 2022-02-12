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
from collections import defaultdict, deque
from thesis.core.encoder import Positions6Max
from thesis.canonical_vectorizer import CanonicalVectorizer
import enum


class ActionSpace(enum.IntEnum):
  """todo: map from RLEnv actions to this discretized version..."""
  FOLD = 0
  CHECK = 1
  CALL = 2
  RAISE_3BB = 3
  RAISE_HALF_POT = 3
  RAISE_POT = 4
  RAISE_2POT = 5
  ALL_IN = 6
  SMALL_BLIND = 7
  BIG_BLIND = 8


class Wrapper:

  def __init__(self, env):
    """
    Args:
        env (PokerEnv subclass instance):   The environment instance to be wrapped
    """
    # assert issubclass(type(env), PokerEnv)
    self.env = env
    self._table = NotImplementedError
    self._player_hands = []

  # _______________________________ directly interact with the env inside the wrapper ________________________________
  def step(self, action):
    """
    Steps the environment from an action of the natural action representation to the environment.

    Returns:
        obs, reward, done, info
    """
    # store action in history buffer
    self._pushback_action(action,
                          player_who_acted=self.env.current_player.seat_id,
                          in_which_stage=self.env.current_round)
    # step environment
    env_obs, rew_for_all_players, done, info = self.env.step(action)

    # call get_current_obs of derived class
    return self._return_obs(env_obs=env_obs, rew_for_all_players=rew_for_all_players, done=done, info=info)

  def step_from_processed_tuple(self, action):
    """
    Steps the environment from a tuple (action, num_chips,).

    Returns:
        obs, reward, done, info
    """
    # store action in history buffer
    self._pushback_action(action,
                          player_who_acted=self.env.current_player.seat_id,
                          in_which_stage=self.env.current_round)
    # step environment
    env_obs, rew_for_all_players, done, info = self.env.step_from_processed_tuple(action)

    # call get_current_obs of derived class
    return self._return_obs(env_obs=env_obs, rew_for_all_players=rew_for_all_players, done=done, info=info)

  def step_raise_pot_frac(self, pot_frac):
    """
    Steps the environment from a fractional pot raise instead of an action as usually specified.

    Returns:
        obs, reward, done, info
    """
    processed_action = (2, self.env.get_fraction_of_pot_raise(
      fraction=pot_frac, player_that_bets=self.env.current_player))
    return self.env.step(processed_action)

  def reset(self, state_dict=None):
    # todo: consider moving this to derived cls
    deck_state_dict = state_dict['deck']
    self._table = state_dict['table']
    self._player_hands = deck_state_dict['hand']
    env_obs, rew_for_all_players, done, info = self.env.reset(deck_state_dict=deck_state_dict)
    return self._return_obs(env_obs=env_obs, rew_for_all_players=rew_for_all_players, done=done, info=info)

  def _return_obs(self, rew_for_all_players, done, info, env_obs=None):
    return self.get_current_obs(env_obs=env_obs), rew_for_all_players, done, info

  # _______________________________ Override to augment observation ________________________________

  def get_current_obs(self, env_obs):
    raise NotImplementedError

  def _pushback_action(self, action, player_who_acted, in_which_stage):
    raise NotImplementedError


class ActionHistory:
  # noinspection PyTypeChecker
  def __init__(self, max_players, max_actions_per_player_per_stage):
    self._max_players = max_players
    self._max_actions_per_player_per_stage = max_actions_per_player_per_stage
    self.deque = {}

    for pos in range(self._max_players):
      # create default dictionary for current player for each stage
      # default dictionary stores only the last two actions per stage per player
      self.deque[Positions6Max(pos)] = defaultdict(
        lambda: deque(maxlen=max_actions_per_player_per_stage),
        keys=['preflop', 'flop', 'turn', 'river'])


class AugmentObservationWrapper(Wrapper):

  def __init__(self, env):
    super().__init__(env=env)
    self.num_players = env.N_SEATS
    self._table = None
    self._rounds = ['preflop', 'flop', 'turn', 'river']
    # self._actions_per_stage = self._init_player_actions()
    self._actions_per_stage = ActionHistory(max_players=6, max_actions_per_player_per_stage=2)
    self._actions_per_stage_discretized = ActionHistory(max_players=6, max_actions_per_player_per_stage=2)
    self._vectorizer = CanonicalVectorizer()
    self._player_who_acted = None

  def get_current_obs(self, env_obs):
    return self._vectorizer.vectorize(env_obs, table=self._table, action_history=self._actions_per_stage,
                                      player_hands=self._player_hands)

  def _discretize(self, action_formatted):
    if action_formatted[0] == 2:  # action is raise
      raise_amt = action_formatted[1]
      # todo compute raise_by using CurrentPot and BB
      raise_by = ActionSpace.RAISE_3BB
      # compute fraction of BB or POT
      action_discretized = 2 + raise_by
    else:  # action is fold or check/call
      action_discretized = action_formatted[0]
    return action_discretized

  def _pushback_action(self, action_formatted, player_who_acted, in_which_stage):
    # todo map action_formatted to ActionSpace
    self._player_who_acted = player_who_acted
    self._actions_per_stage.deque[player_who_acted][
      self._rounds[in_which_stage]].append(action_formatted)

    action_discretized = self._discretize(action_formatted)
    self._actions_per_stage_discretized.deque[player_who_acted][
      self._rounds[in_which_stage]].append(action_discretized)

  @property
  def current_player(self):
    return self.env.current_player
