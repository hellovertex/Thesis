""" This module will
 - read .txt files inside ./data/
 - parse them to create corresponding environment states. """
from typing import List, Tuple, Dict
import re
from core.parser import Parser, PokerEpisode, Action, ActionType, PlayerStack, Blind, PlayerWithCards

# REGEX templates
PLAYER_NAME_TEMPLATE = r'([a-zA-Z0-9_.-@]+\s?[_.-a-zA-Z0-9@]*)'
STARTING_STACK_TEMPLATE = r'\(([$€]\d+.?\d*)\sin chips\)'
MATCH_ANY = r'.*?'  # not the most efficient way, but we prefer readabiliy (parsing is one time job)
POKER_CARD_TEMPLATE = r'[23456789TJQKAjqka][SCDHscdh]'
CURRENCY_SYMBOLS = ['$', '€', '￡']  # only these are currently supported


# ---------------------------- PokerStars-Parser ---------------------------------

class TxtParser(Parser):
    """Reads .txt files with poker games crawled from Pokerstars.com and parses them to
    PokerEpisodes."""

    def __init__(self):
        # todo consider making TxtParser another abstract class and make derived PokerStars-Parser
        self._variant = None
        self._metadata = {'n_total_episodes': 0,
                          'n_total_showdowns': 0,
                          'n_mucks': 0,
                          'n_showdowns_no_mucks': 0}

    def _reset_metadata_counts(self):
        self._metadata = {'n_total_episodes': 0,
                          'n_total_showdowns': 0,
                          'n_mucks': 0,
                          'n_showdowns_no_mucks': 0}

    @property
    def metadata(self):
        return self._metadata

    @staticmethod
    def get_hand_id(episode: str) -> int:
        """Returns the episode number of the current hand played. """
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
    def get_winner(showdown: str) -> Tuple[List[PlayerWithCards], List[PlayerWithCards]]:
        """Return player name of player that won showdown."""
        re_showdown_hands = re.compile(
            rf'Seat \d: {PLAYER_NAME_TEMPLATE}{MATCH_ANY} showed (\[{POKER_CARD_TEMPLATE} {POKER_CARD_TEMPLATE}])')
        re_winner = re.compile(
            rf'Seat \d: {PLAYER_NAME_TEMPLATE}{MATCH_ANY} showed (\[{POKER_CARD_TEMPLATE} {POKER_CARD_TEMPLATE}]) and won')
        showdown_hands = re_showdown_hands.findall(showdown)
        winners = re_winner.findall(showdown)

        # remove whitespaces in name field
        showdown_hands = [PlayerWithCards(name=hand[0].strip(), cards=hand[1])
                          for hand in showdown_hands]
        winners = [PlayerWithCards(name=hand[0].strip(), cards=hand[1])
                   for hand in winners]
        return winners, showdown_hands

    @staticmethod
    def _get_action_type(line: str):
        """Returns either 'fold', 'check_call', or 'raise."""
        default_raise_amount = -1  # for fold, check and call actions
        if 'raises' in line or 'bets' in line:
            pattern = re.compile(r'(\d+\.?\d*)')
            raise_amount = pattern.findall(line)[-1]
            return ActionType.RAISE, raise_amount
        if 'calls' in line or 'checks' in line:
            return ActionType.CHECK_CALL, default_raise_amount
        if 'folds' in line:
            return ActionType.FOLD, default_raise_amount
        raise RuntimeError(f"Could not parse action type from line: \n{line}")

    @staticmethod
    def get_actions(action_history: str, stage: str) -> List[Action]:
        """This is best explained by an example. Consider the string
          '''jimjames32: raises $4 to $6\n
          HHnguyen15: raises $14 to $20\n
          Pierson2323 joins the table at seat #5\n
          poppy20633 joins the table at seat #6\n
          3JackOFF: folds\n
          jimjames32: calls $14'''

          Each valid action follows the pattern {PLAYERNAME}: {action}\n
          So we split each line by ':', and check, which of the splitresults has exactly two elements
          (playername, action).
        """
        possible_actions = [possible_action.split(':') for possible_action in action_history.split('\n')]
        actions = []
        for maybe_action in possible_actions:
            if len(maybe_action) == 2:
                action_type, raise_amount = TxtParser._get_action_type(maybe_action[1])
                action = Action(player_name=maybe_action[0],
                                action_type=action_type,
                                raise_amount=raise_amount,
                                stage=stage)
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
        pattern = re.compile(
            rf"{PLAYER_NAME_TEMPLATE}: posts (small blind|big blind) ([$€]\d+.?\d*)")
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
        """Returns a string with board cards, e.g. '[6h Ts Td 9c Jc]'. """
        summary = episode.split("*** SUMMARY ***")
        pattern = re.compile(r'Board\s(\[.*?])\n')
        return pattern.findall(summary[1])[0]

    def _parse_actions(self, episode: str) -> Dict[str, List[Action]]:
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

        actions_preflop = self.get_actions(hole_cards, stage='preflop')
        actions_flop = self.get_actions(flop, stage='flop')
        actions_turn = self.get_actions(turn, stage='turn')
        actions_river = self.get_actions(river, stage='river')
        as_sequence = actions_preflop + actions_flop + actions_turn + actions_river
        return {'preflop': actions_preflop,
                'flop': actions_flop,
                'turn': actions_turn,
                'river': actions_river,
                'as_sequence': as_sequence}

    @staticmethod
    def get_currency_symbol(episode: str):
        for sbl in CURRENCY_SYMBOLS:
            if sbl in episode:
                return sbl
        raise ValueError("Currency symbol not supported")

    def _parse_episode(self, episode: str, showdown: str):
        """UnderConstruction"""
        hand_id = self.get_hand_id(episode)
        currency_symbol = self.get_currency_symbol(episode)
        winners, showdown_hands = self.get_winner(showdown)
        # blinds = self.get_blinds(episode)
        blinds = [Blind(*blind) for blind in self.get_blinds(episode)]
        btn = self.get_button(episode)
        player_stacks = [PlayerStack(*stack) for stack in self.get_player_stacks(episode)]
        num_players = len(player_stacks)
        btn_idx = self.get_btn_idx(player_stacks, btn)
        board_cards = self.get_board_cards(episode)
        actions_total = self._parse_actions(episode)

        return PokerEpisode(date='',  # todo
                            hand_id=hand_id,
                            variant=self._variant,
                            currency_symbol=currency_symbol,
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
            self._metadata['n_total_episodes'] += 1
            # Only parse hands that went to Showdown stage, i.e. were shown
            if not '*** SHOW DOWN ***' in current:
                continue

            # get showdown
            self._metadata['n_total_showdowns'] += 1
            showdown = self.get_showdown(current)

            # skip if player did not show hand
            if 'mucks' in showdown:
                self._metadata['n_mucks'] += 1
                continue

            self._metadata['n_showdowns_no_mucks'] += 1
            yield self._parse_episode(current, showdown)

    def parse_file(self, file_path):
        # self._reset_metadata_counts()
        self._variant = 'NoLimitHoldem'  # todo parse from filename
        with open(file_path, 'r') as f:  # pylint: disable=invalid-name,unspecified-encoding
            hand_database = f.read()
            hands_played = re.split(r'PokerStars Hand #', hand_database)[1:]
            return self._parse_hands(hands_played)
