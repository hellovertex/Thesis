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
from thesis.core.wrapper import WrapperPokerRL
from thesis.vectorizer_PokerRL import CanonicalVectorizer
import enum


class ActionSpace(enum.IntEnum):
    """Under Construction"""
    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISE_MIN_OR_3BB = 3
    RAISE_HALF_POT = 4
    RAISE_POT = 5
    ALL_IN = 6
    SMALL_BLIND = 7
    BIG_BLIND = 8


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


class ActionHistoryWrapper(WrapperPokerRL):

    def __init__(self, env):
        """
        Args:
            env (PokerEnv subclass instance):   The environment instance to be wrapped
        """
        super().__init__(env=env)
        self._table = None
        self._player_hands = []
        self._rounds = ['preflop', 'flop', 'turn', 'river']
        self._actions_per_stage = ActionHistory(max_players=6, max_actions_per_player_per_stage=2)
        self._actions_per_stage_discretized = ActionHistory(max_players=6, max_actions_per_player_per_stage=2)

    # _______________________________ Overridden ________________________________
    def _before_step(self, action):
        """
        Steps the environment from an action of the natural action representation to the environment.

        Returns:
            obs, reward, done, info
        """
        # store action in history buffer
        self._pushback_action(action,
                              player_who_acted=self.env.current_player.seat_id,
                              in_which_stage=self.env.current_round)

    def _before_reset(self, config=None):
        self._table = config['table']
        self._player_hands = config['deck_state_dict']['hand']

    # _______________________________ Action History ________________________________

    def discretize(self, action_formatted):
      if action_formatted[0] == 2:  # action is raise
        pot_size = self.env.get_all_winnable_money()
        raise_amt = action_formatted[1]
        if raise_amt < pot_size / 2:
          return ActionSpace.RAISE_MIN_OR_3BB
        elif raise_amt < pot_size:
          return ActionSpace.RAISE_HALF_POT
        elif raise_amt < 2 * pot_size:
          return ActionSpace.RAISE_POT
        else:
          return ActionSpace.ALL_IN
      else:  # action is fold or check/call
        action_discretized = action_formatted[0]
      return action_discretized

    def _pushback_action(self, action_formatted, player_who_acted, in_which_stage):
        self._player_who_acted = player_who_acted
        # part of observation
        self._actions_per_stage.deque[player_who_acted][
            self._rounds[in_which_stage]].append(action_formatted)

        action_discretized = self.discretize(action_formatted)
        # for the neural network labels
        self._actions_per_stage_discretized.deque[player_who_acted][
          self._rounds[in_which_stage]].append(action_discretized)

    # _______________________________ Override to Augment observation ________________________________
    def get_current_obs(self, env_obs):
        """Implement this to encode Action History into observation"""
        raise NotImplementedError


class AugmentObservationWrapper(ActionHistoryWrapper):

    def __init__(self, env):
        super().__init__(env=env)
        self._normalization_sum = float(
            sum([s.starting_stack_this_episode for s in self.env.seats])
        ) / self.env.N_SEATS
        self.num_players = env.N_SEATS
        self._table = None
        self._vectorizer = CanonicalVectorizer()
        self._player_who_acted = None

    def get_current_obs(self, env_obs):
        return self._vectorizer.vectorize(env_obs, table=self._table, action_history=self._actions_per_stage,
                                          player_hands=self._player_hands)

    @property
    def current_player(self):
        return self.env.current_player
