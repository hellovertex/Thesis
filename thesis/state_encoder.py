from typing import List, Tuple
import numpy as np
from collections import defaultdict, deque
from core.parser import PokerEpisode, Action, ActionType, PlayerStack
from core.encoder import Encoder
from PokerRL.game.games import NoLimitHoldem
from thesis.core.encoder import PlayerInfo, Positions6Max


class RLStateEncoder(Encoder):
    Observations = List[List]
    Actions_Taken = List[Tuple[int, int]]
    currency_symbol = '$'

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

    def __init__(self, env_builder=None):
        self._env_builder = env_builder

    # noinspection PyTypeChecker
    @staticmethod
    def _init_player_actions(player_info):
        player_actions = {}
        for p_info in player_info:
            # create default dictionary for current player for each stage
            # default dictionary stores only the last two actions per stage per player
            player_actions[p_info.player_name] = defaultdict(
                lambda: deque(maxlen=2), keys=['preflop', 'flop', 'turn', 'river'])
        return player_actions

    @staticmethod
    def roll_position_indices(num_players: int, btn_idx: int) -> np.ndarray:
        """ # Roll position indices, such that each seat is assigned correct position
    # Example: btn_idx=1
    # ==> np.roll([0,1,2], btn_idx) returns [2,0,1]:
    # The first  seat has position index 2, which is BB
    # The second seat has position index 0, which is BTN
    # The third  seat has position index 1, which is SB """
        return np.roll(np.arange(num_players), btn_idx)

    @staticmethod
    def build_all_player_info(player_stacks: List[PlayerStack], rolled_position_indices) -> Tuple[PlayerInfo]:
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

    @staticmethod
    def make_blinds(blinds: List[Tuple[str]], multiply_by: int = 1):
        sb = blinds[0]
        assert sb[1] == 'small blind'
        bb = blinds[1]
        assert bb[1] == 'big blind'
        return int(sb[2].split(RLStateEncoder.currency_symbol)[1]) * multiply_by, \
               int(bb[2].split(RLStateEncoder.currency_symbol)[1]) * multiply_by

    @staticmethod
    def _str_cards_to_list(cards: str):
        """ See example below """
        # '[6h Ts Td 9c Jc]'
        rm_brackets = cards.replace('[', '').replace(']', '')
        # '6h Ts Td 9c Jc'
        card_list = rm_brackets.split(' ')
        # ['6h', 'Ts', 'Td', '9c', 'Jc']
        return card_list

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

        return [[self.DICT_RANK[card[0]], self.DICT_SUITE[card[1]]] for card in card_list]

    def make_player_hands(self, player_info, showdown_hands):
        """Under Construction. """
        name = 3  # index
        position = 1  # index
        assert len(showdown_hands) == 2
        name_0 = showdown_hands[0][0]
        name_1 = showdown_hands[1][0]
        # '[6h Ts]' to ['6h', 'Ts']
        cards_0 = self._str_cards_to_list(showdown_hands[0][1])
        cards_1 = self._str_cards_to_list(showdown_hands[1][1])
        # initialize default hands
        player_hands = [[-127, -127] for _ in range(len(player_info))]

        # overwrite known hands
        for player in player_info:
            if player[name] in [name_0, name_1]:
                # overwrite hand for player 0
                if player[name] == name_0:
                    hand = [[self.DICT_RANK[card[0]], self.DICT_SUITE[card[1]]] for card in cards_0]
                    player_hands[player[position]] = hand
                # overwrite hand for player 1
                else:
                    hand = [[self.DICT_RANK[card[0]], self.DICT_SUITE[card[1]]] for card in cards_1]
                    player_hands[player[position]] = hand
        return player_hands

    @staticmethod
    def build_action(action: Action, multiply_by=100):
        """Under Construction."""
        return action.action_type.value, int(float(action.raise_amount) * multiply_by)

    @staticmethod
    def _init_env(player_info: Tuple[PlayerInfo]):
        """Initializes environment used to generate observations."""
        # sort the player list such button is first, regardless of seat number
        player_info_sorted = np.roll(player_info, player_info[0].position_index, axis=0)
        # get starting stacks, starting with button at index 0
        starting_stack_sizes_list = [int(float(stack) * 100) for stack in player_info_sorted[:, 4]]

        # make args for env
        args = NoLimitHoldem.ARGS_CLS(n_seats=len(player_info),
                                      starting_stack_sizes_list=starting_stack_sizes_list)
        # return env instance
        # todo consider wrapping
        return NoLimitHoldem(is_evaluating=True,
                             env_args=args,
                             lut_holder=NoLimitHoldem.get_lut_holder())

    def _build_cards_state_dict(self, player_info: Tuple[PlayerInfo], episode: PokerEpisode):
        board_cards = self.make_board_cards(episode.board_cards)
        # --- set deck ---
        # cards are drawn without ghost cards, so we simply replace the first 5 cards of the deck
        # with the board cards that we have parsed
        deck = np.empty(shape=(13 * 4, 2), dtype=np.int8)
        deck[:len(board_cards)] = board_cards
        # make hands: np.ndarray(shape=(n_players, 2, 2))
        player_hands = self.make_player_hands(player_info, episode.showdown_hands)
        return {'deck': {'deck_remaining': deck},  # np.ndarray(shape=(52-n_cards*num_players, 2))
                'board': np.full((5, 2), -127),  # np.ndarray(shape=(n_cards, 2))
                'hand': player_hands}

    def _encode_episode(self, episode):
        pass

    def encode_episode(self, episode: PokerEpisode) -> Tuple[Observations, Actions_Taken]:

        # --- Initialize environment --- #
        rolled_position_indices = self.roll_position_indices(episode.num_players, episode.btn_idx)
        player_info = self.build_all_player_info(episode.player_stacks, rolled_position_indices)
        env = self._init_env(player_info)
        # utils
        showdown_players = [info[0] for info in episode.showdown_hands]
        winner_names = [winner[0] for winner in episode.winners]
        # --- set blinds ---
        env.SMALL_BLIND, env.BIG_BLIND = self.make_blinds(episode.blinds, multiply_by=100)

        # --- Reset it with new state_dict --- #
        cards_state_dict = self._build_cards_state_dict(player_info, episode)
        obs, _, done, _ = env.reset(deck_state_dict=cards_state_dict)

        # --- Step Environment with action --- #
        action_sequence = episode.actions_total['as_sequence']
        actions_formatted = [self.build_action(action) for action in action_sequence]
        train_data = []
        labels = []
        it = 0
        while not done:
            action = actions_formatted[it]
            next_to_act = env.current_player.seat_id
            for player in player_info:
                if player.position_index == next_to_act and player.player_name in showdown_players:
                    train_data.append(obs)
                    if player.player_name in winner_names:
                        labels.append(action)
                    else:
                        # replace action call/raise with fold
                        labels.append((ActionType.FOLD.value, -1))
                        # todo make this a little more sophisticated
            obs, _, done, _ = env.step(action)

            it += 1
        return train_data, labels
