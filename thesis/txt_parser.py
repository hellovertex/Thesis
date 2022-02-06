""" This module will
 - read .txt files inside ./data/
 - parse them to create corresponding environment states. """
from time import time
import re
import enum
from typing import NamedTuple, List, Tuple, Dict, Iterable
import numpy as np
from collections import defaultdict, deque
from PokerRL.game.games import NoLimitHoldem



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
currency_symbol = '$'  # or #€


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


# ---------------------------- Parser ---------------------------------
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




class TxtParser(Parser):
    def __init__(self):
        self._variant = None

    @staticmethod
    def get_hand_id(episode: str) -> int:
        pattern = re.compile(r'^(\d+):')
        return int(pattern.findall(episode)[0])

    @staticmethod
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

    @staticmethod
    def get_winner(showdown: str):
        """Return player name of player that won showdown."""
        re_showdown_hands = re.compile(
            rf'Seat \d: {PLAYER_NAME_TEMPLATE}{MATCH_ANY} showed (\[{POKER_CARD_TEMPLATE} {POKER_CARD_TEMPLATE}])')
        re_winner = re.compile(
            rf'Seat \d: {PLAYER_NAME_TEMPLATE}{MATCH_ANY} showed (\[{POKER_CARD_TEMPLATE} {POKER_CARD_TEMPLATE}]) and won')
        showdown_hands = re_showdown_hands.findall(showdown)
        winner = re_winner.findall(showdown)
        # remove whitespaces in name field
        showdown_hands = [(hand[0].strip(), hand[1]) for hand in showdown_hands]
        winner = [(hand[0].strip(), hand[1]) for hand in winner]
        return winner, showdown_hands

    @staticmethod
    def _get_action_type(line: str):
        """Returns either 'fold', 'check_call', or 'raise."""
        default_raise_amount = -1  # for fold, check and call actions
        if 'raises' in line or 'bets' in line:
            raise_amount = line.split(currency_symbol)[1].split()[0]
            return ActionType.RAISE, raise_amount
        elif 'calls' in line or 'checks' in line:
            return ActionType.CHECK_CALL, default_raise_amount
        elif 'folds' in line:
            return ActionType.FOLD, default_raise_amount
        else:
            raise RuntimeError(f"Could not parse action type from line: \n{line}")

    @staticmethod
    def get_actions(stage: str) -> List[Action]:
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
                action_type, raise_amount = TxtParser._get_action_type(maybe_action[1])
                action = Action(player_name=maybe_action[0], action_type=action_type, raise_amount=raise_amount)
                actions.append(action)
        return actions

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def get_board_cards(episode: str):
        summary = episode.split("*** SUMMARY ***")
        pattern = re.compile(r'Board\s(\[.*?])\n')
        return pattern.findall(summary[1])[0]

    def _parse_actions(self, episode: str) -> Dict[str: Action]:
        """Returns a dictionary with actions per stage.
          Args:
            episode: string representation of played episode as gotten from .txt files
          Returns:
            Dictionary with actions per stage:
            {'preflop': actions_preflop,
                'flop': actions_flop,
                'turn': actions_turn,
                'river': actions_river,
                'as_sequence': as_sequence}
        """
        hole_cards = episode.split("*** HOLE CARDS ***")[1].split("*** FLOP ***")[0]
        flop = episode.split("*** FLOP ***")[1].split("*** TURN ***")[0]
        turn = episode.split("*** TURN ***")[1].split("*** RIVER ***")[0]
        river = episode.split("*** RIVER ***")[1].split("*** SHOW DOWN ***")[0]

        actions_preflop = self.get_actions(hole_cards)
        actions_flop = self.get_actions(flop)
        actions_turn = self.get_actions(turn)
        actions_river = self.get_actions(river)
        as_sequence = actions_preflop + actions_flop + actions_turn + actions_river
        return {'preflop': actions_preflop,
                'flop': actions_flop,
                'turn': actions_turn,
                'river': actions_river,
                'as_sequence': as_sequence}

    def _parse_episode(self, episode: str, showdown: str):
        """UnderConstruction"""
        hand_id = self.get_hand_id(episode)
        winners, showdown_hands = self.get_winner(showdown)
        blinds = self.get_blinds(episode)
        btn = self.get_button(episode)
        player_stacks = [PlayerStack(*stack) for stack in self.get_player_stacks(episode)]
        num_players = len(player_stacks)
        btn_idx = self.get_btn_idx(player_stacks, btn)
        board_cards = self.get_board_cards(episode)
        actions_total = self._parse_actions(episode)

        return PokerEpisode(date='',  # todo
                            hand_id=hand_id,
                            variant=self._variant,
                            num_players=num_players,
                            blinds=blinds,
                            player_stacks=player_stacks,
                            btn_idx=btn_idx,
                            board_cards=board_cards,
                            actions_total=actions_total,
                            winners=winners,
                            showdown_hands=showdown_hands)

    def _parse_hands(self, hands_played):
        for current in hands_played:  # c for current_hand
            # Only parse hands that went to Showdown stage, i.e. were shown
            if not '*** SHOW DOWN ***' in current:
                continue
            # get showdown
            showdown = self.get_showdown(current)
            # skip if player did not show hand
            if 'mucks' in showdown:
                continue

            yield self._parse_episode(current, showdown)

    def parse_file(self, file_path):
        self._variant = 'NoLimitHoldem'  # todo parse from filename
        with open(file_path, 'r') as f:  # pylint: disable=invalid-name,unspecified-encoding
            hand_database = f.read()
            hands_played = re.split(r'PokerStars Hand #', hand_database)[1:]
            return self._parse_hands(hands_played)


# ---------------------------- Encoder ---------------------------------

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


# noinspection PyTypeChecker
def _init_player_actions(player_info):
    player_actions = {}
    for p_info in player_info:
        # create default dictionary for current player for each stage
        # default dictionary stores only the last two actions per stage per player
        player_actions[p_info.player_name] = defaultdict(lambda: deque(maxlen=2),
                                                         keys=['preflop', 'flop', 'turn', 'river'])
    return player_actions


def roll_position_indices(num_players: int, btn_idx: int) -> np.ndarray:
    """ # Roll position indices, such that each seat is assigned correct position
    # Example: btn_idx=1
    # ==> np.roll([0,1,2], btn_idx) returns [2,0,1]:
    # The first  seat has position index 2, which is BB
    # The second seat has position index 0, which is BTN
    # The third  seat has position index 1, which is SB """
    return np.roll(np.arange(num_players), btn_idx)


def build_all_player_info(player_stacks: List[PlayerStack], rolled_position_indices):
    """ Docstring """
    # 1. Roll position indices, such that each seat is assigned correct position

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


def make_blinds(blinds: List[Tuple[str]], multiply_by: int = 1):
    sb = blinds[0]
    assert sb[1] == 'small blind'
    bb = blinds[1]
    assert bb[1] == 'big blind'
    return int(sb[2].split(currency_symbol)[1]) * multiply_by, \
           int(bb[2].split(currency_symbol)[1]) * multiply_by


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


def _str_cards_to_list(cards: str):
    """ See example below """
    # '[6h Ts Td 9c Jc]'
    rm_brackets = cards.replace('[', '').replace(']', '')
    # '6h Ts Td 9c Jc'
    card_list = rm_brackets.split(' ')
    # ['6h', 'Ts', 'Td', '9c', 'Jc']
    return card_list


def make_board_cards(board_cards: str):
    """Return 5 cards that we can prepend to the card deck so that the board will be drawn.
      Args:
        board_cards: for example '[6h Ts Td 9c Jc]'
      Returns:
        representation of board_cards that is understood by rl_env
        Example:
    """
    # '[6h Ts Td 9c Jc]' to ['6h', 'Ts', 'Td', '9c', 'Jc']
    card_list = _str_cards_to_list(board_cards)
    assert len(card_list) == 5

    return [[DICT_RANK[card[0]], DICT_SUITE[card[1]]] for card in card_list]


def make_player_hands(player_info, showdown_hands):
    """Under Construction. """
    name = 3  # index
    position = 1  # index
    assert len(showdown_hands) == 2
    name_0 = showdown_hands[0][0]
    name_1 = showdown_hands[1][0]
    # '[6h Ts]' to ['6h', 'Ts']
    cards_0 = _str_cards_to_list(showdown_hands[0][1])
    cards_1 = _str_cards_to_list(showdown_hands[1][1])
    # initialize default hands
    player_hands = [[-127, -127] for _ in range(len(player_info))]

    # overwrite known hands
    for player in player_info:
        if player[name] in [name_0, name_1]:
            # overwrite hand for player 0
            if player[name] == name_0:
                hand = [[DICT_RANK[card[0]], DICT_SUITE[card[1]]] for card in cards_0]
                player_hands[player[position]] = hand
            # overwrite hand for player 1
            else:
                hand = [[DICT_RANK[card[0]], DICT_SUITE[card[1]]] for card in cards_1]
                player_hands[player[position]] = hand
    return player_hands


def build_action(action: tuple):
    """Under Construction."""
    # todo
    return action


def main(f_path: str):
    """Parses hand_database and returns vectorized observations as returned by rl_env."""
    parser = TxtParser()
    parsed_hands = parser.parse_file(f_path)


    for hand in parsed_hands:
        player_stacks = hand.player_stacks
        num_players = hand.num_players
        btn_idx = hand.btn_idx
        actions_total = hand.actions_total
        # todo move to processing unit

        # corresponds to env.reset()
        rolled_position_indices = roll_position_indices(num_players, btn_idx)
        player_info = build_all_player_info(player_stacks, rolled_position_indices)

        player_hands = make_player_hands(player_info, hand.showdown_hands)

        STACK_COLUMN = 4

        # sort the player list such button is first, regardless of seat number
        player_info_sorted = np.roll(player_info, player_info[0].position_index, axis=0)
        starting_stack_sizes_list = [int(float(stack) * 100) for stack in player_info_sorted[:, STACK_COLUMN]]

        # *** Obtain encoded observation *** #
        # --- Create new env for every hand --- #
        args = NoLimitHoldem.ARGS_CLS(n_seats=num_players,
                                      starting_stack_sizes_list=starting_stack_sizes_list)
        env = NoLimitHoldem(is_evaluating=True, env_args=args, lut_holder=NoLimitHoldem.get_lut_holder())
        # --- Reset it with new state_dict --- #
        board_cards = make_board_cards(hand.board_cards)
        # verify using env.cards2str(board_cards)
        # --- set blinds ---
        sb, bb = make_blinds(hand.blinds, multiply_by=100)
        env.SMALL_BLIND = sb
        env.BIG_BLIND = bb
        # --- set deck ---
        # cards are drawn without ghost cards, so we simply replace the first 5 cards of the deck
        # with the board cards that we have parsed
        deck = np.empty(shape=(13 * 4, 2), dtype=np.int8)
        deck[:len(board_cards)] = board_cards
        # set hands

        cards_state_dict = {'deck': {'deck_remaining': deck},  # np.ndarray(shape=(52-n_cards*num_players, 2))
                            'board': np.full((5, 2), -127),  # np.ndarray(shape=(n_cards, 2))
                            'hand': player_hands}  # np.ndarray(shape=(n_players, 2, 2))
        obs, reward, done, info = env.reset(deck_state_dict=cards_state_dict)

        # todo: step environment with actions_per_stage and player_info
        action_sequence = actions_total['as_sequence']
        actions_formatted = [build_action(action) for action in action_sequence]
        # while not done: env.step(next(actions_formatted))
        # todo: check how obs is normalized to avoid small floats

        # *** Observation Augmentation *** #
        # raise vs bet: raise only in preflop stage, bet after preflop
        actions_per_stage = _init_player_actions(player_info)

        for stage, actions in actions_total.items():
            for action in actions:
                # noinspection PyTypeChecker
                actions_per_stage[action.player_name][stage].append((action.action_type, action.raise_amount))

        # todo: augment inside env wrapper
        # --- Append last 8 moves per player --- #
        # --- Append all players hands --- #
        debug = 1


if __name__ == "__main__":
    """
    EXAMPLE EPISODE:
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

    """
    EXAMPLE OBSERVATION FOR 2 PLAYERS
                          ante:   0.0
                 small_blind:   0.05
                   big_blind:   0.1
                   min_raise:   0.2
                     pot_amt:   0.0
               total_to_call:   0.1
        last_action_how_much:   0.1
          last_action_what_0:   0.0
          last_action_what_1:   1.0
          last_action_what_2:   0.0
           last_action_who_0:   1.0
           last_action_who_1:   0.0
           last_action_who_2:   0.0
                p0_acts_next:   0.0
                p1_acts_next:   1.0
                p2_acts_next:   0.0
               round_preflop:   1.0
                  round_flop:   0.0
                  round_turn:   0.0
                 round_river:   0.0
                  side_pot_0:   0.0
                  side_pot_1:   0.0
                  side_pot_2:   0.0
                    stack_p0:   0.9
                 curr_bet_p0:   0.1
  has_folded_this_episode_p0:   0.0
                 is_allin_p0:   0.0
       side_pot_rank_p0_is_0:   0.0
       side_pot_rank_p0_is_1:   0.0
       side_pot_rank_p0_is_2:   0.0
                    stack_p1:   0.95
                 curr_bet_p1:   0.05
  has_folded_this_episode_p1:   0.0
                 is_allin_p1:   0.0
       side_pot_rank_p1_is_0:   0.0
       side_pot_rank_p1_is_1:   0.0
       side_pot_rank_p1_is_2:   0.0
                    stack_p2:   0.9
                 curr_bet_p2:   0.1
  has_folded_this_episode_p2:   0.0
                 is_allin_p2:   0.0
       side_pot_rank_p2_is_0:   0.0
       side_pot_rank_p2_is_1:   0.0
       side_pot_rank_p2_is_2:   0.0
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
  None
  EXAMPLE DECK:
  {'deck_remaining': array([[10,  2],
         [ 4,  3],
         [ 8,  1],
         [ 2,  3],
         [ 9,  2],
         [11,  3],
         [ 6,  0],
         [ 7,  2],
         [10,  1],
         [ 6,  2],
         [ 6,  1],
         [ 7,  0],
         [11,  0],
         [ 5,  1],
         [ 3,  3],
         [ 7,  3],
         [ 1,  0],
         [ 1,  1],
         [10,  0],
         [ 5,  3],
         [12,  0],
         [ 0,  2],
         [ 2,  1],
         [ 8,  0],
         [12,  3],
         [ 8,  2],
         [ 2,  2],
         [ 4,  0],
         [10,  3],
         [11,  2],
         [ 3,  2],
         [ 5,  0],
         [ 2,  0],
         [ 1,  3],
         [ 9,  0],
         [ 0,  3],
         [ 9,  1],
         [ 7,  1],
         [ 5,  2],
         [12,  2],
         [ 3,  0],
         [ 1,  2],
         [ 6,  3],
         [ 9,  3],
         [ 4,  1],
         [ 0,  0]], dtype=int8)}
  simply put them in order from BTN to CU
  cards will then be dealt starting with BTN
  starting_stack_sizes_list 
  
  
  everything including the hole cards can be built from load_state_dict using EnvDictIdxs
    """
    F_PATH = '../data/Atalante-1-2-USD-NoLimitHoldem-PokerStarsPA-1-16-2022.txt'
    main(F_PATH)
