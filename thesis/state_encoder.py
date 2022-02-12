from typing import List, Tuple, Dict, Optional, NamedTuple
import numpy as np
from collections import defaultdict, deque
from core.parser import PokerEpisode, Action, ActionType, PlayerStack
from core.encoder import Encoder
from PokerRL.game.games import NoLimitHoldem
from thesis.core.encoder import PlayerInfo, Positions6Max
from thesis.core.wrapper import AugmentedEnvBuilder
from PokerRL.game.Poker import Poker
from enum import Enum
from thesis.canonical_vectorizer import CanonicalVectorizer

Table = Tuple[PlayerInfo]

DICT_RANK = {'': -127,
             '2': 0,
             '3': 1,
             '4': 2,
             '5': 3,
             '6': 4,
             '7': 5,
             '8': 6,
             '9': 7,
             'T': 8,
             'J': 9,
             'Q': 10,
             'K': 11,
             'A': 12}

DICT_SUITE = {'': -127,
              'h': 0,
              'd': 1,
              's': 2,
              'c': 3}


# class Table6Max(NamedTuple):
#   BTN: PlayerInfo
#   SB: PlayerInfo
#   BB: Optional[PlayerInfo]
#   UTG: Optional[PlayerInfo]
#   MP: Optional[PlayerInfo]
#   CO: Optional[PlayerInfo]


class RLStateEncoder(Encoder):
  Observations = List[List]
  Actions_Taken = List[Tuple[int, int]]
  currency_symbol = '$'

  def __init__(self, env_builder_cls=None):
    self._env_builder_cls = env_builder_cls
    self._env_builder: Optional[AugmentedEnvBuilder] = None

  def _get_wrapped_env(self, table: Tuple[PlayerInfo], multiply_by=100):
    """Initializes environment used to generate observations.
    Assumes Btn is at index 0."""
    # get starting stacks, starting with button at index 0
    stacks = [player.stack_size for player in table]
    starting_stack_sizes_list = [int(float(stack) * multiply_by) for stack in stacks]

    # make args for env
    args = NoLimitHoldem.ARGS_CLS(n_seats=len(table),
                                  starting_stack_sizes_list=starting_stack_sizes_list)
    # return wrapped env instance
    self._env_builder = self._env_builder_cls(env_cls=NoLimitHoldem, env_args=args)
    env = NoLimitHoldem(is_evaluating=True,
                        env_args=self._env_builder.env_args,
                        lut_holder=NoLimitHoldem.get_lut_holder())

    return self._env_builder.get_new_wrapper(is_evaluating=True, init_from_env=env, table=table)



  @staticmethod
  def _str_cards_to_list(cards: str):
    """ See example below """
    # '[6h Ts Td 9c Jc]'
    rm_brackets = cards.replace('[', '').replace(']', '')
    # '6h Ts Td 9c Jc'
    card_list = rm_brackets.split(' ')
    # ['6h', 'Ts', 'Td', '9c', 'Jc']
    return card_list

  @staticmethod
  def make_blinds(blinds: List[Tuple[str]], multiply_by: int = 1):
    sb = blinds[0]
    assert sb[1] == 'small blind'
    bb = blinds[1]
    assert bb[1] == 'big blind'
    return int(sb[2].split(RLStateEncoder.currency_symbol)[1]) * multiply_by, \
           int(bb[2].split(RLStateEncoder.currency_symbol)[1]) * multiply_by

  def make_board_cards(self, board_cards: str):
    """Return 5 cards that we can prepend to the card deck so that the board will be drawn.
  Args:
    board_cards: for example '[6h Ts Td 9c Jc]'
  Returns:
    representation of board_cards that is understood by rl_env
    Example:
"""
    # '[6h Ts Td 9c Jc]' to ['6h', 'Ts', 'Td', '9c', 'Jc']
    card_list = self._str_cards_to_list(board_cards)
    assert len(card_list) == 5

    return [[DICT_RANK[card[0]], DICT_SUITE[card[1]]] for card in card_list]

  def make_showdown_hands(self, table, showdown):
    """Under Construction. """
    # initialize default hands
    player_hands = [Poker.CARD_NOT_DEALT_TOKEN_2D for _ in range(len(table))]

    # overwrite known hands
    for seat in table:
      for final_player in showdown:
        if seat.player_name == final_player.name:
          # '[6h Ts]' to ['6h', 'Ts']
          showdown_cards = self._str_cards_to_list(final_player.cards)
          # ['6h', 'Ts'] to [[5,3], [5,0]]
          hand = [[DICT_RANK[card[0]], DICT_SUITE[card[1]]] for card in showdown_cards]
          # overwrite [[-127,127],[-127,-127]] with [[5,3], [5,0]]
          player_hands[int(seat.position_index)] = hand
    return player_hands

  @staticmethod
  def build_action(action: Action, multiply_by=100):
    """Under Construction."""
    return action.action_type.value, int(float(action.raise_amount) * multiply_by)

  def _build_cards_state_dict(self, table: Tuple[PlayerInfo], episode: PokerEpisode):
    board_cards = self.make_board_cards(episode.board_cards)
    # --- set deck ---
    # cards are drawn without ghost cards, so we simply replace the first 5 cards of the deck
    # with the board cards that we have parsed
    deck = np.empty(shape=(13 * 4, 2), dtype=np.int8)
    deck[:len(board_cards)] = board_cards
    # make hands: np.ndarray(shape=(n_players, 2, 2))
    player_hands = self.make_showdown_hands(table, episode.showdown_hands)
    board = np.full((5, 2), Poker.CARD_NOT_DEALT_TOKEN_1D, dtype=np.int8)
    return {'deck': {'deck_remaining': deck},  # np.ndarray(shape=(52-n_cards*num_players, 2))
            'board': board,  # np.ndarray(shape=(n_cards, 2))
            'hand': player_hands}

  @staticmethod
  def _roll_position_indices(num_players: int, btn_idx: int) -> np.ndarray:
    """ Roll position indices, such that each seat is assigned correct position.
    Args:
      btn_idx: seat index (not seat number) of seat that is currently the Button.
                Seats can be ["Seat 1", "Seat3", "Seat 5"]. If "Seat 5" is the Button,
                btn_idx=2
      num_players: Number of players currently at the table (not max. players).
    Returns: Assignment of position indices to seat numbers.

    Example: btn_idx=1
        # ==> np.roll([0,1,2], btn_idx) returns [2,0,1]:
        # The first  seat has position index 2, which is BB
        # The second seat has position index 0, which is BTN
        # The third  seat has position index 1, which is SB
    """
    # np.roll([0,1,2,3], 1) returns [3,0,1,2]  <== Button is at index 1 now
    return np.roll(np.arange(num_players), btn_idx)

  def make_table(self, episode: PokerEpisode) -> Table:
    """Docstring """
    # Roll position indices, such that each seat is assigned correct position
    rolled_position_indices = self._roll_position_indices(episode.num_players, episode.btn_idx)

    # init {'BTN': None, 'SB': None,..., 'CO': None}
    player_info: Dict[str, PlayerInfo] = dict.fromkeys(
      [pos.name for pos in Positions6Max])  # [:episode.num_players])

    # build PlayerInfo for each player
    for i, info in enumerate(episode.player_stacks):
      seat_number = int(info.seat_display_name[5])
      player_name = info.player_name
      stack_size = float(info.stack[1:])
      position_index = rolled_position_indices[i]
      position = Positions6Max(position_index).name
      player_info[position] = PlayerInfo(seat_number,  # 2
                                         position_index,  # 0
                                         position,  # 'BTN'
                                         player_name,  # 'JoeSchmoe Billy'
                                         stack_size)

    # Seat indices such that button is first, regardless of seat number
    players_ordered_starting_with_button = [v for v in player_info.values()]
    return tuple(players_ordered_starting_with_button)

  @staticmethod
  def _encode_env_transitions(env_obs, actions) -> Tuple[Observations, Actions_Taken]:

    # todo: obs + self._actions_per_stage + player_hands + zero padding
    # vectorized = self._vec.vectorize(obs)
    return env_obs, actions  # todo implement

  def _simulate_environment(self, env, episode, cards_state_dict, table, cbs_action=[]):
    """Docstring"""
    showdown_players = [player.name for player in episode.showdown_hands]
    winner_names = [winner.name for winner in episode.winners]

    action_sequence = episode.actions_total['as_sequence']

    obs, _, done, _ = env.reset(deck_state_dict=cards_state_dict)

    # --- Step Environment with action --- #
    observations = []
    actions = []
    it = 0
    while not done:
      action = action_sequence[it]
      action_formatted = self.build_action(action)
      # store up to two actions per player per stage
      # self._actions_per_stage[action.player_name][action.stage].append(action_formatted)
      next_to_act = env.current_player.seat_id
      for player in table:
        if player.position_index == next_to_act and player.player_name in showdown_players:
          player_hands = [[-127, -127] for _ in range(len(table))]
          player_hands[next_to_act] = env.env.seats[next_to_act].hand
          observations.append(obs)
          if player.player_name in winner_names:
            actions.append(action_formatted)
          else:
            # replace action call/raise with fold
            actions.append((ActionType.FOLD.value, -1))
      obs, _, done, _ = env.step(action_formatted)
      it += 1
    return observations, actions

  def encode_episode(self, episode: PokerEpisode) -> Tuple[Observations, Actions_Taken]:
    """Runs environment with steps from PokerEpisode.
    Returns observations and corresponding actions of players that made it to showdown."""
    # utils
    table = self.make_table(episode)

    # Initialize environment for simulation of PokerEpisode
    # todo: pass env_cls as argument (N_BOARD_CARDS etc. gets accessible)
    wrapped_env = self._get_wrapped_env(table)
    wrapped_env.SMALL_BLIND, wrapped_env.BIG_BLIND = self.make_blinds(episode.blinds, multiply_by=100)
    cards_state_dict = self._build_cards_state_dict(table, episode)

    # Collect observations and actions, observations are possibly augmented
    observations, actions = self._simulate_environment(env=wrapped_env,
                                                       episode=episode,
                                                       cards_state_dict=cards_state_dict,
                                                       table=table,
                                                       cbs_action=wrapped_env.pushback_action)

    # Vectorize collected observations and actions for supervised learning
    return self._encode_env_transitions(observations, actions)
