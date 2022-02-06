from typing import NamedTuple, Iterable, List, Dict
import enum


class PlayerStack(NamedTuple):
    """Player Stacks as parsed from the textfiles.
    For example: PlayerStack('Seat 1', 'jimjames32', '$82 ')
    """
    seat_display_name: str
    player_name: str
    stack: str


class ActionType(enum.IntEnum):
    FOLD = 0
    CHECK_CALL = 1
    RAISE = 2


class Action(NamedTuple):
    """If the current bet is 30, and the agent wants to bet 60 chips more, the action should be (2, 90)"""
    player_name: str
    action_type: ActionType
    raise_amount: float = -1


class PokerEpisode(NamedTuple):
    """UnderConstruction"""
    date: str
    hand_id: int
    variant: str
    num_players: int
    blinds: list
    player_stacks: List[PlayerStack]
    btn_idx: int
    board_cards: str
    actions_total: Dict[str, List[Action]]
    winners: list
    showdown_hands: list


class Parser:
    """ Abstract Parser Interface. All parsers should be derived from this base class
    and implement the method "parse_file"."""

    def parse_file(self, file_path) -> Iterable[PokerEpisode]:
        """Reads file that stores played poker hands and returns and iterator over the played hands.
        Args:
          file_path: path to the database file that contains hands crawled from a specific poker website.
        Returns: An Iterable of PokerEpisodes.

        """
        raise NotImplementedError
