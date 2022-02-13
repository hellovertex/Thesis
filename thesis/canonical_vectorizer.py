from thesis.core.vectorizer import Vectorizer
from PokerEnv.PokerRL.game.Poker import Poker
import numpy as np


class CanonicalVectorizer(Vectorizer):
  """Docstring"""

  # todo write vectorizer, such that it can be associated with exactly one env class
  def __init__(self, env, max_players=6, n_ranks=13, n_suits=4, n_board_cards=5, n_hand_cards=2):
    # --- Utils --- #
    self._env = env
    self._max_players = max_players
    self._n_ranks = n_ranks
    self._n_suits = n_suits
    self._n_board_cards = n_board_cards
    self._n_hand_cards = n_hand_cards
    self._n_stages = len(['preflop', 'flop', 'turn', 'river'])
    self._player_hands = None
    self._action_history = None
    self._table = None
    # btn_idx is equal to current player offset, since button is at index 0 inside environment
    # but we encode observation such that player is at index 0
    self._btn_idx = self._env.BTN_POS
    max_actions_per_stage_per_player = 2
    max_actions = max_actions_per_stage_per_player * self._n_stages + self._max_players
    self._bits_stats_per_player_original = len(['stack', 'curr_bet', 'has_folded', 'is_all_in']) \
                                  + len(['side_pot_rank_p0_is_']) * self._env.N_SEATS
    self._bits_stats_per_player = len(['stack', 'curr_bet', 'has_folded', 'is_all_in']) \
                                  + len(['side_pot_rank_p0_is_']) * self._max_players
    self._bits_per_card = n_ranks + n_suits  # 13 ranks + 4 suits
    self._bits_per_action = self._max_players \
                            + len(['fold', 'check/call', 'bet/raise']) \
                            + len(['last_action_how_much'])
    # --- Observation Bits --- #
    self._bits_table = len(['ante',
                            'small_blind',
                            'big_blind',
                            'min_raise',
                            'pot_amt',
                            'total_to_call'])
    self._bits_next_player = self._max_players
    self._bits_stage = self._n_stages
    self._bits_side_pots = self._max_players
    self._bits_player_stats = self._bits_stats_per_player * self._max_players
    self._bits_player_stats_original = self._bits_stats_per_player_original * self._env.N_SEATS
    self._bits_board = self._bits_per_card * n_board_cards  # 3 cards flop, 1 card turn, 1 card river
    self._bits_player_hands = self._max_players * n_hand_cards * self._bits_per_card
    self._bits_action_history = max_actions * self._bits_per_action

    # --- Offsets --- #
    self._start_table = 0
    self._start_next_player = self._bits_table
    self._start_stage = self._start_next_player + self._bits_next_player
    self._start_side_pots = self._start_stage + self._bits_stage
    self._start_player_stats = self._start_side_pots + self._bits_side_pots
    self._start_board = self._start_player_stats + self._bits_player_stats
    self._start_player_hands = self._start_board + self._bits_board
    self._start_action_history = self._start_player_hands + self._bits_player_hands
    self.offset = None

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
    """

    self.offset = 0 + self._bits_table
    assert self.offset == self._start_next_player
    # copy unchanged
    self._obs[0:self.offset] = obs[0:self.offset]

  def encode_next_player(self, obs):
    """Example:
        p0_acts_next:   0.0
        p1_acts_next:   1.0
        p2_acts_next:   0.0
      This encodes the btn_idx, because we moved self to index 0.
      So we do not have to encode self._btn_idx explicitly.
    """
    self.offset += self._bits_next_player
    assert self.offset == self._start_stage
    # original obs indices
    start_orig = self._env.obs_idx_dict['p0_acts_next']
    end_orig = start_orig + self._env.N_SEATS
    # extract from original observation
    bits = obs[start_orig:end_orig]
    # zero padding
    bits = np.resize(bits, self._max_players)
    # copy from original observation with zero padding
    self._obs[self._start_next_player:self.offset] = bits

  def encode_stage(self, obs):
    """Example:
       round_preflop:   1.0
          round_flop:   0.0
          round_turn:   0.0
         round_river:   0.0
    """
    self.offset += self._bits_stage
    assert self.offset == self._start_side_pots
    # original obs indices
    start_orig = self._env.obs_idx_dict['round_preflop']
    end_orig = start_orig + self._n_stages
    # extract from original observation
    bits = obs[start_orig:end_orig]
    # zero padding is not necessary
    # copy from original observation without zero padding
    self._obs[self._start_stage:self.offset] = bits
  
  def encode_side_pots(self, obs):
    """Example:
        side_pot_0:   0.0
        side_pot_1:   0.0
        side_pot_2:   0.0
    """
    self.offset += self._bits_side_pots
    assert self.offset == self._start_player_stats
    # original obs indices
    start_orig = self._env.obs_idx_dict['side_pot_0']
    end_orig = start_orig + self._env.N_SEATS
    # extract from original observation
    bits = obs[start_orig:end_orig]
    # move self to index 0
    bits = np.roll(bits, -self._env.current_player.seat_id)
    # zero padding
    bits = np.resize(bits, self._max_players)
    # copy from original observation with zero padding
    self._obs[self._start_side_pots:self.offset] = bits

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
    self.offset += self._bits_player_stats
    assert self.offset == self._start_board
    # original obs indices
    start_orig = self._env.obs_idx_dict['stack_p0']
    end_orig = start_orig + self._bits_player_stats_original
    # extract from original observation
    bits = obs[start_orig:end_orig]
    # move self to index 0
    bits = np.roll(bits, -self._env.current_player.seat_id*self._bits_stats_per_player_original)
    # zero padding
    bits = np.resize(bits, self._bits_player_stats)
    # copy from original observation with zero padding
    self._obs[self._start_player_stats:self.offset] = bits

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
    self.offset += self._bits_board
    assert self.offset == self._start_player_hands
    # original obs indices
    start_orig = self._env.obs_idx_dict['0th_board_card_rank_0']
    end_orig = start_orig + self._bits_board
    # extract from original observation
    bits = obs[start_orig:end_orig]
    # zero padding is not necessary
    # copy from original observation without zero padding
    self._obs[self._start_board:self.offset] = bits

  def encode_player_hands(self, obs):
    """Example:"""
    self.offset += self._bits_player_hands
    assert self.offset == self._start_action_history
    # move own cards to index 0
    roll_by = -self._env.current_player.seat_id
    rolled_cards = np.roll(self._player_hands, roll_by).reshape(-1, self._n_hand_cards)
    # rolled_cards = [[ 5  3], [ 5  0], [12  0], [ 9  1], [ -127  -127], [ -127  -127]]
    # replace NAN with 0
    rolled_cards[np.where(rolled_cards == Poker.CARD_NOT_DEALT_TOKEN_1D)] = 0
    # rolled_cards = [[ 5  3], [ 5  0], [12  0], [ 9  1], [ 0  0], [ 0  0]]

    # initialize hand_bits to 0
    card_bits = self._n_ranks + self._n_suits
    hand_bits = [0] * self._n_hand_cards * self._env.N_SEATS * card_bits
    # overwrite one_hot card_bits
    for n_card, card in enumerate(rolled_cards):
      offset = card_bits * n_card
      # set rank
      hand_bits[card[0] + offset] = 1
      # set suit
      hand_bits[card[1] + offset + self._n_ranks] = 1
    # zero padding
    hand_bits = np.resize(hand_bits, self._bits_player_hands)
    self._obs[self._start_player_hands:self.offset] = hand_bits

  def encode_action_history(self, obs):
    """Example:"""
    self.offset += self._bits_action_history
    assert self.offset == self._obs_len

  def vectorize(self, obs, action_history=None, player_hands=None, table=None):
    self._player_hands = player_hands
    self._action_history = action_history
    self._table = table
    # todo consider passing obs_idx_dict instead of using self._env
    # use table information to do the zero padding and the index switch
    # obs contains already actions_per_stage and player_hands
    self.encode_table(obs)
    self.encode_next_player(obs)
    self.encode_stage(obs)
    self.encode_side_pots(obs)
    self.encode_player_stats(obs)
    self.encode_board(obs)
    self.encode_player_hands(obs)
    self.encode_action_history(obs)
    return np.resize(obs, (159,))
