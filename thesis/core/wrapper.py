# todo: augment state
# todo: optional, change FOLD replacements to be less aggressive
# todo from action-tuple and observation, construct discrete action (i.e. discretize raise amount)
from PokerRL.game.Poker import Poker
from PokerRL.game.PokerRange import PokerRange
from PokerRL.game.games import DiscretizedNLHoldem, DiscretizedNLLeduc
from PokerRL.game.wrappers import VanillaEnvBuilder, HistoryEnvBuilder
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
