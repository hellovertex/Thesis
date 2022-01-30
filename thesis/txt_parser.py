""" This module will
 - read .txt files inside ./data/
 - parse them to create corresponding environment states. """
from time import time
import re
import enum
from typing import NamedTuple, List, Tuple
import numpy as np
from collections import defaultdict, deque

class Positions6Max(enum.IntEnum):
  """Positions as found in the literature, for a table with at most 6 Players.
  BTN for Button, SB for Small Blind, etc...
  """
  BTN = 0
  SB = 1
  BB = 2
  UTG = 3
  MP = 4
  CO = 5


class Positions9Max(enum.IntEnum):
  """Positions as found in the literature, for a table with at most 9 Players.
  BTN for Button, SB for Small Blind, etc...
  """
  BTN = 0
  SB = 1
  BB = 2
  UTG = 3
  UTG1 = 4
  UTG2 = 5
  MP = 6
  MP1 = 7
  CO = 8


class PlayerInfo(NamedTuple):
  """Player information as parsed from the textfiles.
  For example: PlayerInfo(seat_number=1, position_index=0, position='BTN',
  player_name='jimjames32', stack_size=82.0)
  """
  seat_number: int
  position_index: int  # 0 for BTN, 1 for SB, 2 for BB, etc.
  position: str  # c.f. Positions6Max or Positions9Max
  player_name: str
  stack_size: float


class PlayerStack(NamedTuple):
  """Player Stacks as parsed from the textfiles.
  For example: PlayerStack('Seat 1', 'jimjames32', '$82 ')
  """
  seat_display_name: str
  player_name: str
  stack: str


# REGEX templates
PLAYER_NAME_TEMPLATE = r'([a-zA-Z0-9]+\s?[a-zA-Z0-9]*)'
STARTING_STACK_TEMPLATE = r'\(([$€]\d+.?\d*)\sin chips\)'
MATCH_ANY = r'.*?'  # not the most efficient way, but we prefer readabiliy (parsing is one time job)
POKER_CARD_TEMPLATE = r'[23456789TJQKAjqka][SCDHscdh]'


def get_button(episode: str) -> int:
  """Returns the buttons seat number as displayed to user.
      Args:
          :episode string representation of played episode as gotten from .txt files
      Returns:
          button: int representing the seat number as displayed to user
  """
  ptn_button = re.compile(r"Seat #\d is the button")
  # ptn_button2 = re.compile(r"Seat #(\d) is the button")
  # button2 = int(ptn_button2.findall(s)[0])
  button = ptn_button.findall(episode)[0][6]
  return int(button)  # 1-indexed


def get_player_stacks(line: str):
  """Returns stacks for each player.
      Args:
          :episode string representation of played episode as gotten from .txt files
      Returns:
          Example: [('Seat 1', 'jimjames32', '$82 '),
                    ('Seat 2', 'HHnguyen15', '$96.65'),
                    ('Seat 4', 'kjs609', '$200 ')]
  """
  # pattern = re.compile(r"(Seat \d): ([a-zA-Z0-9]+\s?[a-zA-Z0-9]*)\s\(([$€]\d+.?\d*)\sin chips\)")
  # pattern = re.compile(rf"(Seat \d): {PLAYER_NAME_TEMPLATE}\s\(([$€]\d+.?\d*)\sin chips\)")
  pattern = re.compile(rf"(Seat \d): {PLAYER_NAME_TEMPLATE}\s{STARTING_STACK_TEMPLATE}")
  return pattern.findall(line)


def get_blinds(episode: str) -> List[Tuple[str]]:
  """Returns blinds for current hand.
  Args:
      :episode string representation of played episode as gotten from .txt files
  Returns:
      Example: [('HHnguyen15', 'small blind', '$1'), ('kjs609', 'big blind', '$2')]
  """
  # pattern = re.compile(r"([a-zA-Z0-9]+): posts (small blind|big blind) ([$€]\d+.?\d*)")
  pattern = re.compile(rf"{PLAYER_NAME_TEMPLATE}: posts (small blind|big blind) ([$€]\d+.?\d*)")
  return pattern.findall(episode)


def get_btn_idx(player_stacks: List[PlayerStack], btn_seat_num: int):
  """Returns seat index (not seat number) of seat that is currently the Button.
  Seats can be ["Seat 1", "Seat3", "Seat 5"]. If "Seat 5" is the Button,
  btn_idx=2 will be returned.
      Args:
          :player_stacks list of player info after parsing .txt files
      Returns:
          int index of button
  """
  # determine btn_idx
  for i, player_stack in enumerate(player_stacks):
    if int(player_stack.seat_display_name[5]) == btn_seat_num:
      return i
  raise RuntimeError(
    "Button index could not be determined. Guess we have to do more debugging...")


def _roll_position_indices(num_players: int, btn_idx: int) -> np.ndarray:
  """ # Roll position indices, such that each seat is assigned correct position
  # Example: btn_idx=1
  # ==> np.roll([0,1,2], btn_idx) returns [2,0,1]:
  # The first  seat has position index 2, which is BB
  # The second seat has position index 0, which is BTN
  # The third  seat has position index 1, which is SB """
  return np.roll(np.arange(num_players), btn_idx)


def build_all_player_info(player_stacks: List[PlayerStack], num_players, btn_idx):
  """ Docstring """
  # 1. roll seats position assignment depending on where button sits
  rolled_position_indices = _roll_position_indices(num_players, btn_idx)
  player_infos = []
  # build PlayerInfo for each player
  for i, info in enumerate(player_stacks):
    seat_number = int(info.seat_display_name[5])
    player_name = info.player_name
    stack_size = float(info.stack[1:])
    position_index = rolled_position_indices[i]
    position = Positions6Max(position_index).name
    player_infos.append(PlayerInfo(seat_number,  # 2
                                   position_index,  # 0
                                   position,  # 'BTN'
                                   player_name,  # 'JoeSchmoe Billy'
                                   stack_size))  # 82.45
  return player_infos


def get_showdown(episode: str):
  """Return True if the current episode does not have a showdown.
  Args:
      episode: string representation of played episode as gotten from .txt files.
      Episode is assumed to contain showdown.
  Returns:
  """
  hands_played = re.split(r'\*\*\* SHOW DOWN \*\*\*', episode)
  assert len(hands_played) == 2, \
    f"Splitting showdown string went wrong: splits are {hands_played} "
  return hands_played[1]


def get_winner(showdown: str):
  """Return player name of player that won showdown."""
  re_showdown_hands = re.compile(
    rf'Seat \d: {PLAYER_NAME_TEMPLATE}{MATCH_ANY} showed (\[{POKER_CARD_TEMPLATE} {POKER_CARD_TEMPLATE}])')
  re_winner = re.compile(
    rf'Seat \d: {PLAYER_NAME_TEMPLATE}{MATCH_ANY} showed (\[{POKER_CARD_TEMPLATE} {POKER_CARD_TEMPLATE}]) and won')
  showdown_hands = re_showdown_hands.findall(showdown)
  winner = re_winner.findall(showdown)
  return showdown_hands, winner


class Action(enum.IntEnum):
  FOLD: 0
  CHECK_CALL: 1
  RAISE: 2


def get_player_actions(stage: str):
  """This is best explained by an example. Consider the string
    '''jimjames32: raises $4 to $6\n
    HHnguyen15: raises $14 to $20\n
    Pierson2323 joins the table at seat #5\n
    poppy20633 joins the table at seat #6\n
    3JackOFF: folds\n
    jimjames32: calls $14'''

    Each valid action follows the pattern {PLAYERNAME}: {action}\n
    So we split each line by ':', and check, which of the splitresults has exactly two elements (playername, action)
  """
  possible_actions = [possible_action.split(':') for possible_action in stage.split('\n')]
  actions = []
  for maybe_action in possible_actions:
    if len(maybe_action) == 2:
      actions.append(maybe_action)
  return actions


def main(f_path: str):
  """Parses hand_database and returns vectorized observations as returned by rl_env."""
  t_0 = time()
  with open(f_path, 'r', ) as f:  # pylint: disable=invalid-name,unspecified-encoding
    hand_database = f.read()
    print(f'loading took {time() - t_0} seconds...')
    hands_played = re.split(r'PokerStars Hand #', hand_database)[1:]

    # todo fill boards cards
    # todo: split between hole cards flop turn river
    for current in hands_played:  # c for current_hand
      # Only parse hands that went to Showdown stage, i.e. were shown
      if not '*** SHOW DOWN ***' in current:
        continue
      # get showdown
      showdown = get_showdown(current)
      # skip if player did not show hand
      if 'mucks' in showdown:
        continue
      # get hero
      # todo deal with split pot scenario when there are two heroes
      hero, showdown_hands = get_winner(showdown)
      blinds = get_blinds(current)
      btn = get_button(current)
      player_stacks = [PlayerStack(*stack) for stack in get_player_stacks(current)]
      num_players = len(player_stacks)
      btn_idx = get_btn_idx(player_stacks, btn)
      # corresponds to env.reset()
      player_info = build_all_player_info(player_stacks,
                                          num_players,
                                          btn_idx)
      hole_cards = current.split("*** HOLE CARDS ***")[1].split("*** FLOP ***")[0]
      flop = current.split("*** FLOP ***")[1].split("*** TURN ***")[0]
      turn = current.split("*** TURN ***")[1].split("*** RIVER ***")[0]
      river = current.split("*** RIVER ***")[1].split("*** SHOW DOWN ***")[0]

      # raise vs bet: raise only in preflop stage, bet after preflop
      player_actions = defaultdict(lambda: deque(maxlen=2))  # store only the last 2 actions per stage
      for p_info in player_info:
        player_actions[p_info.player_name].append('a1')
        player_actions[p_info.player_name].append('a2')
        player_actions[p_info.player_name].append('a3')  # only a2 and a3 will have survived

      # todo: fill player_actions{
      #  'pid1': [preflop1, preflop2, flop1, flop2, river1, ...],
      #  'pid2': [preflop1, preflop2, flop1, flop2, river1, ...],...}
      actions_preflop = get_player_actions(hole_cards)
      actions_flop = get_player_actions(flop)
      actions_turn = get_player_actions(turn)
      actions_river = get_player_actions(river)
      print(f'parsing one episode took {time() - t_0} seconds...')
      debug = 1


if __name__ == "__main__":
  """
  Mobster365: folds
  sziget37: folds
  Kraliya: raises $0.10 to $0.20
  sts1981: folds
  alaver24: raises $0.50 to $0.70
  milkimen: folds
  BUICKBOY2: calls $0.65
  PlayLaughGro: folds
  Kraliya: raises $0.50 to $1.20
  alaver24: calls $0.50
  BUICKBOY2: raises $1.53 to $2.73 and is all-in
  Kraliya: raises $1.53 to $4.26
  alaver24: folds
  Uncalled bet ($1.53) returned to Kraliya
  ... showdown yadayadayada
  """
  F_PATH = '../data/Atalante-1-2-USD-NoLimitHoldem-PokerStarsPA-1-16-2022.txt'
  main(F_PATH)
