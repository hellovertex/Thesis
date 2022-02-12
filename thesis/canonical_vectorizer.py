from thesis.core.vectorizer import Vectorizer
import numpy as np


class CanonicalVectorizer(Vectorizer):
  """Docstring"""

  # todo write vectorizer, such that it can be associated with exactly one env class
  def __init__(self, max_players=6, n_ranks=13, n_suits=4, n_board_cards=5, n_hand_cards=2):
    # --- Utils --- #
    self._max_players = max_players
    n_stages = len(['preflop', 'flop', 'turn', 'river'])
    max_actions_per_stage_per_player = 2
    max_actions = max_actions_per_stage_per_player * n_stages + self._max_players
    self._bits_stats_per_player = len(['stack', 'curr_bet', 'has_folded', 'is_all_in']) \
                                  + len(['side_pot_rank_p0_is_']) * self._max_players
    self._bits_per_card = n_ranks + n_suits  # 13 ranks + 4 suits

    # todo thermomether encoding for action_what
    self._bits_per_action = self._max_players \
                            + len(['fold', 'check/call', 'bet/raise']) \
                            + len(['last_action_how_much'])
    # --- Observation Bits --- #
    self._bits_table = len(['ante',
                            'small_blind',
                            'big_blind',
                            'min_raise',
                            'pot_amt',
                            'total_to_call',
                            ] + ['is_button'] * self._max_players)
    self._bits_next_player = self._max_players
    self._bits_stage = n_stages
    self._bits_side_pots = self._max_players
    self._bits_player_stats = self._bits_stats_per_player * self._max_players
    self._bits_board = self._bits_per_card * n_board_cards  # 3 cards flop, 1 card turn, 1 card river
    self._bits_player_hands = self._max_players * n_hand_cards * self._bits_per_card
    self._bits_action_history = max_actions * self._bits_per_action

    # --- Offsets --- #
    self._offset_table = 0
    self._offset_next_player = self._bits_table
    self._offset_stage = self._offset_next_player + self._bits_next_player
    self._offset_side_pots = self._offset_stage + self._bits_stage
    self._offset_player_stats = self._offset_side_pots + self._bits_side_pots
    self._offset_board = self._offset_player_stats + self._bits_player_stats
    self._offset_player_hands = self._offset_board + self._bits_board
    self._offset_action_history = self._offset_player_hands + self._bits_player_hands

    # --- Number of features --- #
    self._obs_len = self._bits_table \
                    + self._bits_next_player \
                    + self._bits_stage \
                    + self._bits_side_pots \
                    + self._bits_player_stats \
                    + self._bits_board \
                    + self._bits_player_hands \
                    + self._bits_action_history
    self._obs = np.zeros(self._obs_len)

  def encode_table(self, obs):
    """Example:
                ante:   0.0
         small_blind:   0.05
           big_blind:   0.1
           min_raise:   0.2
             pot_amt:   0.0
       total_to_call:   0.1
         is_button_0:   0.0
         is_button_1:   1.0
         is_button_2:   0.0
    """
    # copy unchanged
    self._obs[self._offset_table:self._offset_next_player] = obs[self._offset_table:self._offset_next_player]

  def encode_next_player(self, obs, num_players):
    """Example:
        p0_acts_next:   0.0
        p1_acts_next:   1.0
        p2_acts_next:   0.0
    """
    offset = self._offset_next_player
    bits_from_obs = np.array(obs[offset:offset + num_players])
    # obs only has num_players <= max_players bits here,
    # so we pad the non existing players with zeros
    bits_padded = bits_from_obs.resize(self._bits_next_player)
    self._obs[self._offset_next_player:self._offset_stage] = bits_padded

  def encode_stage(self, obs):
    """Example:
       round_preflop:   1.0
          round_flop:   0.0
          round_turn:   0.0
         round_river:   0.0
    """
    # todo get actual offsets for current obs
    # self.obs[self._offset_stage:self._offset_side_pots] = obs[self._offset_stage:self._offset_side_pots]

  def encode_side_pots(self, obs):
    """Example:
       round_preflop:   1.0
          round_flop:   0.0
          round_turn:   0.0
         round_river:   0.0
    """
    return []

  def encode_player_stats(self, obs):
    """Example:
    stack_p0:   0.9
             curr_bet_p0:   0.1
has_folded_this_episode_p0:   0.0
             is_allin_p0:   0.0
   side_pot_rank_p0_is_0:   0.0
   side_pot_rank_p0_is_...:   0.0
   side_pot_rank_p0_is_n:   0.0

                stack_p1:   0.95
             curr_bet_p1:   0.05
has_folded_this_episode_p1:   0.0
             is_allin_p1:   0.0
   side_pot_rank_p1_is_0:   0.0
   side_pot_rank_p1_is_...:   0.0
   side_pot_rank_p1_is_n:   0.0
   """
    return []

  def encode_board(self, obs):
    """Example:
    0th_board_card_rank_0:   0.0
   0th_board_card_rank_1:   0.0
   0th_board_card_rank_2:   0.0
   0th_board_card_rank_3:   0.0
   0th_board_card_rank_4:   0.0
   0th_board_card_rank_5:   0.0
   0th_board_card_rank_6:   0.0
   0th_board_card_rank_7:   0.0
   0th_board_card_rank_8:   0.0
   0th_board_card_rank_9:   0.0
  0th_board_card_rank_10:   0.0
  0th_board_card_rank_11:   0.0
  0th_board_card_rank_12:   0.0
   0th_board_card_suit_0:   0.0
   0th_board_card_suit_1:   0.0
   0th_board_card_suit_2:   0.0
   0th_board_card_suit_3:   0.0
   1th_board_card_rank_0:   0.0
   1th_board_card_rank_1:   0.0
   1th_board_card_rank_2:   0.0
   1th_board_card_rank_3:   0.0
   1th_board_card_rank_4:   0.0
   1th_board_card_rank_5:   0.0
   1th_board_card_rank_6:   0.0
   1th_board_card_rank_7:   0.0
   1th_board_card_rank_8:   0.0
   1th_board_card_rank_9:   0.0
  1th_board_card_rank_10:   0.0
  1th_board_card_rank_11:   0.0
  1th_board_card_rank_12:   0.0
   1th_board_card_suit_0:   0.0
   1th_board_card_suit_1:   0.0
   1th_board_card_suit_2:   0.0
   1th_board_card_suit_3:   0.0
   2th_board_card_rank_0:   0.0
   2th_board_card_rank_1:   0.0
   2th_board_card_rank_2:   0.0
   2th_board_card_rank_3:   0.0
   2th_board_card_rank_4:   0.0
   2th_board_card_rank_5:   0.0
   2th_board_card_rank_6:   0.0
   2th_board_card_rank_7:   0.0
   2th_board_card_rank_8:   0.0
   2th_board_card_rank_9:   0.0
  2th_board_card_rank_10:   0.0
  2th_board_card_rank_11:   0.0
  2th_board_card_rank_12:   0.0
   2th_board_card_suit_0:   0.0
   2th_board_card_suit_1:   0.0
   2th_board_card_suit_2:   0.0
   2th_board_card_suit_3:   0.0
   3th_board_card_rank_0:   0.0
   3th_board_card_rank_1:   0.0
   3th_board_card_rank_2:   0.0
   3th_board_card_rank_3:   0.0
   3th_board_card_rank_4:   0.0
   3th_board_card_rank_5:   0.0
   3th_board_card_rank_6:   0.0
   3th_board_card_rank_7:   0.0
   3th_board_card_rank_8:   0.0
   3th_board_card_rank_9:   0.0
  3th_board_card_rank_10:   0.0
  3th_board_card_rank_11:   0.0
  3th_board_card_rank_12:   0.0
   3th_board_card_suit_0:   0.0
   3th_board_card_suit_1:   0.0
   3th_board_card_suit_2:   0.0
   3th_board_card_suit_3:   0.0
   4th_board_card_rank_0:   0.0
   4th_board_card_rank_1:   0.0
   4th_board_card_rank_2:   0.0
   4th_board_card_rank_3:   0.0
   4th_board_card_rank_4:   0.0
   4th_board_card_rank_5:   0.0
   4th_board_card_rank_6:   0.0
   4th_board_card_rank_7:   0.0
   4th_board_card_rank_8:   0.0
   4th_board_card_rank_9:   0.0
  4th_board_card_rank_10:   0.0
  4th_board_card_rank_11:   0.0
  4th_board_card_rank_12:   0.0
   4th_board_card_suit_0:   0.0
   4th_board_card_suit_1:   0.0
   4th_board_card_suit_2:   0.0
   4th_board_card_suit_3:   0.0
   """
    return []

  def encode_player_hands(self, obs):
    """Example:"""
    return []

  def encode_action_history(self, obs):
    """Example:"""
    return []

  def vectorize(self, obs, action_history=None, player_hands=None, table=None):
    # use table information to do the zero padding and the index switch
    # obs contains already actions_per_stage and player_hands
    # vectorized_obs = self.encode_table(obs) \
    #                  + self.encode_next_player(obs) \
    #                  + self.encode_stage(obs) \
    #                  + self.encode_side_pots(obs) \
    #                  + self.encode_player_stats(obs) \
    #                  + self.encode_board(obs) \
    #                  + self.encode_player_hands(obs) \
    #                  + self.encode_action_history(obs)
    return obs