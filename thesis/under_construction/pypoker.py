"""Python backend for Poker environment code. Poker refers to No-Limit Hold Em here. """
# pylint: disable=useless-object-inheritance
import enum
from typing import List

CARD_CHARS = ['2S', '2C', '2D', '2H', '3S', '3C', '3D', '3H', '4S', '4C', '4D', '4H', '5S', '5C', '5D', '5H', '6S',
              '6C', '6D', '6H', '7S', '7C', '7D', '7H', '8S', '8C', '8D', '8H', '9S', '9C', '9D', '9H', 'TS', 'TC',
              'TD', 'TH', 'JS', 'JC', 'JD', 'JH', 'QS', 'QC', 'QD', 'QH', 'KS', 'KC', 'KD', 'KH', 'AS', 'AC', 'AD',
              'AH']


class PokerCard(object):
    """Poker card with a value and a suite.

    Possible combinations of values = "23456789TJQKA" and suites = "SCDH".
    """

    def __init__(self, value=None, suite=None):
        self._value = value
        self._suite = suite
        self._cid = None if not (value and suite) else CARD_CHARS.index(self._value + self._suite)

    def value(self):
        return self._value

    def suite(self):
        return self._suite

    def cid(self):
        return self._cid

    def __str__(self):
        if self.valid():
            return self._value + self._suite
        else:
            return "XX"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.value == other.value() and self.suite == other.suite()

    def valid(self):
        return self._value and self._suite


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
    def __init__(self, move: PokerMoveType):
        self._move = move

    def to_int(self):
        return self._move.value


class PokerHistoryItem(object):
    pass


class PokerEndOfGameType(enum.IntEnum):
    NOT_FINISHED = 0
    END_HIDDEN = 1
    SHOWDOWN = 2


class PokerTable(object):
    """todo consider using this for PlayerCycle methods such as
    - deactivate_current
    - deactivate_player
    - next_dealer
    - update_alive
    todo and then possibly using this inside PokerState
    """


class PokerState(object):
    """Current environment state for an active Poker game.

      The game is turn-based, with only one active agent at a time..
      """

    def __init__(self, game):
        self._game = game
        self._player_hands = [list() for _ in range(self._game.num_players())]
        self._player_stacks = [self._game.initial_stack_size() for _ in range(self._game.num_players())]

    def observation(self, player_id):
        return None

    def cur_player(self):
        """Returns index of next player to act."""
        return -1  # todo

    def apply_move(self, move):
        pass  # todo

    def player_hands(self):
        return self._player_hands

    def player_hand(self, seat):
        return self._player_hands[seat]

    def player_stacks(self):
        return self._player_stacks

    def set_player_stack(self, value, seat):
        self._player_stacks[seat] = value


class AgentObservationType(enum.IntEnum):
    """Possible agent observation types.
      STANDARD is similar to what a human sees when playing at a table.
      SEER is like STANDARD with the additional knowledge of the other players hands.
    """
    STANDARD = 0
    SEER = 1


class PokerGame(object):
    """Game parameters describing a specific instance of Poker."""

    def __init__(self, params):
        """Creates a PokerGame object.

            Args:
              params: is a dictionary of parameters and their values.

            Possible parameters include
            "players": 2 <= number of players <= 9
            "seed": random number seed. -1 to use system random device to get seed.
            "observation_type": int AgentObservationType.
            """
        self._params = params  # todo
        self._num_players = self._params["players"]
        self._initial_stack_size = 100 if 'initial_stack_size' not in self._params \
            else self._params['initial_stack_size']
        self._hand_size = 2 if 'hand_size' not in self._params else self._params['hand_size']

    def num_players(self):
        """Returns the number of players in the game."""
        return self._num_players  # todo

    def hand_size(self):
        return self._hand_size

    def initial_stack_size(self):
        return self._initial_stack_size


class PokerObservation(object):
    """Player's observed view of an environment PokerState.

      The main differences are that 1) other player's cards are not visible, and
      2) a player does not know their own player index (seat) so that all player
      indices are described relative to the observing player (or equivalently,
      that from the player's point of view, they are always player index 0).
      """

    def __init__(self, state, game, player):
        self._observation = None  # todo remove, bc its only used in encoder
        self._state = None
        self._game = None
        self._player = None
        self._default = -1  # todo remove

    def cur_player_offset(self):
        """Returns the player index of the acting player, relative to observer."""
        return self._default  # todo

    def num_players(self):
        """Returns the number of players in the game."""
        return self._default  # todo

    def deck_size(self):
        """Returns number of cards left in the deck."""
        return self._default  # todo

    def legal_moves(self):
        """Returns list of legal moves for observing player.

        List is empty if cur_player() != 0 (observer is not currently acting).
        """
        moves = []
        for i in range(len(PokerMoveType)):  # todo
            move = PokerMoveType(i)
            moves.append(PokerMove(move))
        return moves

    def _obs_get_hand_card(self, seat, idx) -> PokerCard:  # todo
        return self._state.player_hand(seat)[idx]

    def observed_hands(self) -> List[List[PokerCard]]:
        """Returns a list of all hands.

         The observing players hands are always shown. The other player's hands are not shown unless AgentObservationType.SEER.
        """
        hand_list = []
        for seat in range(self.num_players()):
            player_hand = []
            for i in range(self._game.hand_size):
                player_card = self._obs_get_hand_card(seat, i)
                player_hand.append(player_card)
            hand_list.append(player_hand)
        return hand_list

    def card_range_knowledge(self):
        return [list() for _ in range(self.num_players())]  # todo


class ObservationEncoderType(enum.IntEnum):
    CANONICAL = 0
    DICT_VALUES = 1


class ObservationEncoder(object):
    def __init__(self, game, enc_type=ObservationEncoderType.DICT_VALUES):
        self._game = game
        self._enc_type = enc_type

    def shape(self):
        # todo
        pass

    def encode(self, observation):
        # todo
        return observation
