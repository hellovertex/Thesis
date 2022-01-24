"""Python backend for Poker environment code. Poker refers to No-Limit Hold Em here. """
import enum


class PokerCard(object):
    pass


class PokerMoveType(enum.IntEnum):
    """ Allowed actions """
    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISE_3BB = 3
    RAISE_HALF_POT = 4
    RAISE_POT = 5
    RAISE_2POT = 6
    ALL_IN = 7
    SMALL_BLIND = 8
    BIG_BLIND = 9


class PokerMove(object):
    pass


class PokerHistoryItem(object):
    pass


class PokerEndOfGameType(enum.IntEnum):
    NOT_FINISHED = 0
    END_HIDDEN = 1
    SHOWDOWN = 2


class PokerState(object):
    """Current environment state for an active Poker game.

      The game is turn-based, with only one active agent at a time..
      """
    pass


class AgentObservationType(enum.IntEnum):
    """ Possible agent observation types.
      STANDARD is similar to what a human sees when playing at a table.
      SEER is like STANDARD with the additional knowledge of the other players hands.
      """
    STANDARD = 0
    SEER = 1


class PokerGame(object):
    pass


class PokerObservation(object):
    pass


class ObservationEncoderType(enum.IntEnum):
    CANONICAL = 0
    DICT_VALUES = 1


class ObservationEncoder(object):
    def __init__(self, game, enc_type=ObservationEncoderType.DICT_VALUES):
        self._game = game
        self._enc_type = enc_type

    def shape(self):
        pass

    def encode(self, observation):
        pass
