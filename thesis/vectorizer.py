import enum
import numpy as np
from collections import defaultdict, deque
from typing import NamedTuple, List
from txt_parser import PlayerStack

class Positions6Max(enum.IntEnum):
  """Positions as in the literature, for a table with at most 6 Players.
  BTN for Button, SB for Small Blind, etc...
  """
  BTN = 0
  SB = 1
  BB = 2
  UTG = 3  # UnderTheGun
  MP = 4  # Middle Position
  CO = 5  # CutOff


class Positions9Max(enum.IntEnum):
  """Positions as in the literature, for a table with at most 9 Players.
  BTN for Button, SB for Small Blind, etc...
  """
  BTN = 0
  SB = 1
  BB = 2
  UTG = 3  # UnderTheGun
  UTG1 = 4
  UTG2 = 5
  MP = 6  # Middle Position
  MP1 = 7
  CO = 8  # CutOff


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


class Vectorizer:
  def vectorize(self):
    raise NotImplementedError


class RLEnvVectorizer(Vectorizer):
  def __init__(self):
    self._default = None

  # noinspection PyTypeChecker
  @staticmethod
  def _init_player_actions(player_info):
    player_actions = {}
    for p_info in player_info:
      # create default dictionary for current player for each stage
      # default dictionary stores only the last two actions per stage per player
      player_actions[p_info.player_name] = defaultdict(lambda: deque(maxlen=2),
                                                       keys=['preflop', 'flop', 'turn', 'river'])
    return player_actions


  @staticmethod
  def _roll_position_indices(num_players: int, btn_idx: int) -> np.ndarray:
    """ # Roll position indices, such that each seat is assigned correct position
    # Example: btn_idx=1
    # ==> np.roll([0,1,2], btn_idx) returns [2,0,1]:
    # The first  seat has position index 2, which is BB
    # The second seat has position index 0, which is BTN
    # The third  seat has position index 1, which is SB """
    return np.roll(np.arange(num_players), btn_idx)

  @staticmethod
  def build_all_player_info(player_stacks: List[PlayerStack], num_players, btn_idx):
    """ Docstring """
    # 1. roll seats position assignment depending on where button sits
    rolled_position_indices = RLEnvVectorizer._roll_position_indices(num_players, btn_idx)
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
    return tuple(player_infos)
