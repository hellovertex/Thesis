""" This module will
 - read .txt files inside ./data/
 - parse them to create corresponding environment states.
 - The states are encoded as in neuron_poker.gym_env.env.HoldemTable  """
from time import time
import re
import enum
from typing import NamedTuple, List, Tuple
import numpy as np


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


def get_button(line: str) -> int:
    """Returns the buttons seat number as displayed to user.
        Args:
            :line string representation of hand as gotten from .txt files
        Returns:
            button: int representing the seat number as displayed to user
    """
    ptn_button = re.compile(r"Seat #\d is the button")
    # ptn_button2 = re.compile(r"Seat #(\d) is the button")
    # button2 = int(ptn_button2.findall(s)[0])
    button = ptn_button.findall(line)[0][6]
    return int(button)  # 1-indexed


def get_player_stacks(line: str):
    """Returns stacks for each player.
        Args:
            :line string representation of hand as gotten from .txt files
        Returns:
            Example: [('Seat 1', 'jimjames32', '$82 '),
                      ('Seat 2', 'HHnguyen15', '$96.65'),
                      ('Seat 4', 'kjs609', '$200 ')]
    """
    pattern = re.compile(r"(Seat \d): ([a-zA-Z0-9]+) \(([$€]\d+.?\d*)")
    return pattern.findall(line)


def get_blinds(line: str) -> List[Tuple[str]]:
    """Returns blinds for current hand.
    Args:
        :line string representation of hand as gotten from .txt files
    Returns:
        Example: [('HHnguyen15', 'small blind', '$1'), ('kjs609', 'big blind', '$2')]
    """
    pattern = re.compile(r"([a-zA-Z0-9]+): posts (small blind|big blind) ([$€]\d+.?\d*)")
    return pattern.findall(line)


def get_btn_idx(player_stacks: List[PlayerStack], btn_seat_num: int):
    """Returns seat index (not seat number) of seat that is currently the Button.
    Seats can be ["Seat 1", "Seat3", "Seat 5"]. If "Seat 5" has the Button,
    btn_idx=2 will be returned.
        Args:
            :player_stacks list of player info as gotten from .txt files
        Returns:
            Example: [('HHnguyen15', 'small blind', '$1'), ('kjs609', 'big blind', '$2')]
    """
    # determine btn_idx
    for i, player_stack in enumerate(player_stacks):
        if int(player_stack.seat_display_name[5]) == btn_seat_num:
            return i
    raise RuntimeError(
        "Button index could not be determined. Guess we have to do more debugging...")


def build_all_player_info(player_stacks: List[PlayerStack], num_players, btn_idx):
    """ Docstring """
    # todo: docstring
    rolled_position_indices = np.roll(np.arange(num_players), btn_idx)
    player_infos = []
    for i, info in enumerate(player_stacks):
        seat_number = int(info[0][5])
        display_name = info[1]
        stack_size = float(info[2][1:])
        position_index = rolled_position_indices[i]
        position = Positions6Max(position_index).name  # parse from btn assignment
        player_infos.append(PlayerInfo(seat_number,
                                       position_index,
                                       position,
                                       display_name,
                                       stack_size))
    return player_infos


def main(f_path: str):
    """Parses hand_database and returns vectorized observations as returned by rl_env."""
    t_0 = time()
    with open(f_path, 'r',) as f:  #pylint: disable=invalid-name,unspecified-encoding
        hand_database = f.read()
        print(f'loading took {time() - t_0} seconds...')
        hands_played = re.split(r'PokerStars Hand #', hand_database)

        for current in hands_played:  # c for current_hand
            if current == '':
                continue
            print(current, type(current))
            blinds = get_blinds(current)
            btn = get_button(current)
            player_stacks = [PlayerStack(*stack) for stack in get_player_stacks(current)]
            num_players = len(player_stacks)
            btn_idx = get_btn_idx(player_stacks, btn)
            all_player_info = build_all_player_info(player_stacks, num_players, btn_idx)
            # up until *** HOLE CARDS *** we gather all information from env.reset()
            # *** HOLE CARDS ***
            # *** FLOP ***
            # break


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
