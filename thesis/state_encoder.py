from typing import List, Tuple, Callable, Optional
import numpy as np
from collections import defaultdict, deque
from core.parser import PokerEpisode, Action, ActionType, PlayerStack
from core.encoder import Encoder
from PokerRL.game.games import NoLimitHoldem
from thesis.core.encoder import PlayerInfo, Positions6Max
# from PokerEnv.PokerRL.game._.EnvWrapperBuilderBase import EnvWrapperBuilderBase
from thesis.core.wrapper import AugmentedEnvBuilder


class Vectorizer:

    def __init__(self, max_players=6, n_ranks=13, n_suits=4, n_board_cards=5, n_hand_cards=2):
        # --- Utils --- #
        self._max_players = max_players
        n_stages = len(['preflop', 'flop', 'turn', 'river'])
        max_actions_per_stage_per_player = 2
        max_actions = max_actions_per_stage_per_player * n_stages + self._max_players
        self._bits_stats_per_player = len(['stack', 'curr_bet', 'has_folded', 'is_all_in']) \
                                      + len(['side_pot_rank_p0_is_']) * self._max_players
        self._bits_per_card = n_ranks + n_suits  # 13 ranks + 4 suits
        self._bits_per_action = self._max_players \
                                + len(['fold', 'check/call', 'bet/raise']) \
                                + len(['last_action_how_much'])
        # --- Observation Bits --- #
        self._bits_table = len(['ante',
                                'small_blind',
                                'big_blind',
                                'min_raise',
                                'pot_amt',
                                'total_to_call'])
        self._bits_next_player = self._max_players
        self._bits_stage = n_stages
        self._bits_side_pots = self._max_players
        self._bits_player_stats = self._bits_stats_per_player * self._max_players
        self._bits_board = self._bits_per_card * n_board_cards  # 3 cards flop, 1 card turn, 1 card river
        self._bits_player_hands = self._max_players * n_hand_cards * self._bits_per_card
        self._bits_action_history = max_actions * self._bits_per_action

        # --- Offsets --- #
        self._offset_table = 0
        self._offset_next_player = self._bits_table
        self._offset_stage = self._offset_next_player + self._bits_next_player
        self._offset_side_pots = self._offset_stage + self._bits_stage
        self._offset_player_stats = self._offset_side_pots + self._bits_side_pots
        self._offset_board = self._offset_player_stats + self._bits_player_stats
        self._offset_player_hands = self._offset_board + self._bits_board
        self._offset_action_history = self._offset_player_hands + self._bits_player_hands

        # --- Number of features --- #
        self._obs_len = self._bits_table \
                        + self._bits_next_player \
                        + self._bits_stage \
                        + self._bits_side_pots \
                        + self._bits_player_stats \
                        + self._bits_board \
                        + self._bits_player_hands \
                        + self._bits_action_history
        self._obs = np.zeros(self._obs_len)

    def encode_table(self, obs):
        """Example:
                    ante:   0.0
             small_blind:   0.05
               big_blind:   0.1
               min_raise:   0.2
                 pot_amt:   0.0
           total_to_call:   0.1
        """
        # copy unchanged
        self._obs[self._offset_table:self._offset_next_player] = obs[self._offset_table:self._offset_next_player]

    def encode_next_player(self, obs, num_players):
        """Example:
            p0_acts_next:   0.0
            p1_acts_next:   1.0
            p2_acts_next:   0.0
        """
        offset = self._offset_next_player
        bits_from_obs = np.array(obs[offset:offset + num_players])
        # obs only has num_players <= max_players bits here,
        # so we pad the non existing players with zeros
        bits_padded = bits_from_obs.resize(self._bits_next_player)
        self._obs[self._offset_next_player:self._offset_stage] = bits_padded

    def encode_stage(self, obs):
        """Example:
           round_preflop:   1.0
              round_flop:   0.0
              round_turn:   0.0
             round_river:   0.0
        """
        # todo get actual offsets for current obs
        # self.obs[self._offset_stage:self._offset_side_pots] = obs[self._offset_stage:self._offset_side_pots]

    def encode_side_pots(self, obs):
        """Example:
           round_preflop:   1.0
              round_flop:   0.0
              round_turn:   0.0
             round_river:   0.0
        """
        return []

    def encode_player_stats(self, obs):
        """Example:
        stack_p0:   0.9
                 curr_bet_p0:   0.1
  has_folded_this_episode_p0:   0.0
                 is_allin_p0:   0.0
       side_pot_rank_p0_is_0:   0.0
       side_pot_rank_p0_is_...:   0.0
       side_pot_rank_p0_is_n:   0.0

                    stack_p1:   0.95
                 curr_bet_p1:   0.05
  has_folded_this_episode_p1:   0.0
                 is_allin_p1:   0.0
       side_pot_rank_p1_is_0:   0.0
       side_pot_rank_p1_is_...:   0.0
       side_pot_rank_p1_is_n:   0.0
       """
        return []

    def encode_board(self, obs):
        """Example:
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
       """
        return []

    def encode_player_hands(self, obs):
        """Example:"""
        return []

    def encode_action_history(self, obs):
        """Example:"""
        return []

    def vectorize(self, obs):
        vectorized_obs = self.encode_table(obs) \
                         + self.encode_next_player(obs) \
                         + self.encode_stage(obs) \
                         + self.encode_side_pots(obs) \
                         + self.encode_player_stats(obs) \
                         + self.encode_board(obs) \
                         + self.encode_player_hands(obs) \
                         + self.encode_action_history(obs)


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

    def __init__(self, env_builder_cls=None, vectorizer: Vectorizer=None):
        self._env_builder_cls = env_builder_cls
        self._env_builder: Optional[AugmentedEnvBuilder] = None
        self._actions_per_stage = None
        self._default_player_hands = None
        self._vec = vectorizer

    def _get_wrapped_env(self, player_info: Tuple[PlayerInfo]):
        """Initializes environment used to generate observations."""
        # sort the player list such button is first, regardless of seat number
        player_info_sorted = np.roll(player_info, player_info[0].position_index, axis=0)
        # get starting stacks, starting with button at index 0
        starting_stack_sizes_list = [int(float(stack) * 100) for stack in player_info_sorted[:, 4]]

        # make args for env
        args = NoLimitHoldem.ARGS_CLS(n_seats=len(player_info),
                                      starting_stack_sizes_list=starting_stack_sizes_list)
        # return wrapped env instance
        self._env_builder = self._env_builder_cls(env_cls=NoLimitHoldem, env_args=args)
        env = NoLimitHoldem(is_evaluating=True,
                            env_args=self._env_builder.env_args,
                            lut_holder=NoLimitHoldem.get_lut_holder())

        return self._env_builder.get_new_wrapper(is_evaluating=True, init_from_env=env)

    # noinspection PyTypeChecker
    def _init_player_actions(self, player_info):
        player_actions = {}
        for p_info in player_info:
            # create default dictionary for current player for each stage
            # default dictionary stores only the last two actions per stage per player
            player_actions[p_info.player_name] = defaultdict(
                lambda: deque(maxlen=2), keys=['preflop', 'flop', 'turn', 'river'])
        self._actions_per_stage = player_actions
        return player_actions

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
            if player.player_name in [name_0, name_1]:
                # overwrite hand for player 0
                if player.player_name == name_0:
                    hand = [[self.DICT_RANK[card[0]], self.DICT_SUITE[card[1]]] for card in cards_0]
                    player_hands[player.position_index] = hand
                # overwrite hand for player 1
                else:
                    hand = [[self.DICT_RANK[card[0]], self.DICT_SUITE[card[1]]] for card in cards_1]
                    player_hands[player[position]] = hand
        return player_hands

    @staticmethod
    def build_action(action: Action, multiply_by=100):
        """Under Construction."""
        return action.action_type.value, int(float(action.raise_amount) * multiply_by)

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

    @staticmethod
    def _roll_position_indices(num_players: int, btn_idx: int) -> np.ndarray:
        """ # Roll position indices, such that each seat is assigned correct position
    # Example: btn_idx=1
    # ==> np.roll([0,1,2], btn_idx) returns [2,0,1]:
    # The first  seat has position index 2, which is BB
    # The second seat has position index 0, which is BTN
    # The third  seat has position index 1, which is SB """
        return np.roll(np.arange(num_players), btn_idx)

    def build_all_player_info(self, episode: PokerEpisode) -> Tuple[PlayerInfo]:
        """ Docstring """
        # 1. Roll position indices, such that each seat is assigned correct position
        rolled_position_indices = self._roll_position_indices(episode.num_players, episode.btn_idx)
        player_infos = []
        # build PlayerInfo for each player
        for i, info in enumerate(episode.player_stacks):
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
    def _encode_observation(env_obs: List, stage: str, action: Action):
        return env_obs  # todo implement

    def encode_episode(self, episode: PokerEpisode) -> Tuple[Observations, Actions_Taken]:

        # utils
        player_info = self.build_all_player_info(episode)
        showdown_players = [info[0] for info in episode.showdown_hands]
        winner_names = [winner[0] for winner in episode.winners]

        # --- Initialize environment --- #
        wrapped_env = self._get_wrapped_env(player_info)
        wrapped_env.SMALL_BLIND, wrapped_env.BIG_BLIND = self.make_blinds(episode.blinds, multiply_by=100)
        cards_state_dict = self._build_cards_state_dict(player_info, episode)
        obs, _, done, _ = wrapped_env.reset(deck_state_dict=cards_state_dict)

        # --- Step Environment with action --- #
        action_sequence = episode.actions_total['as_sequence']
        actions_formatted = [self.build_action(action) for action in action_sequence]
        train_data = []
        labels = []
        it = 0
        self._init_player_actions(player_info=player_info)
        while not done:
            action = action_sequence[it]
            action_label = actions_formatted[it]
            # store up to two actions per player per stage
            self._actions_per_stage[action.player_name][action.stage].append(action_label)
            next_to_act = wrapped_env.current_player.seat_id

            for player in player_info:
                if player.position_index == next_to_act and player.player_name in showdown_players:
                    player_hands = [[-127, -127] for _ in range(len(player_info))]
                    player_hands[next_to_act] = wrapped_env.env.seats[next_to_act].hand
                    # todo: obs + self._actions_per_stage + player_hands + zero padding
                    # vectorized = self._vec.vectorize(obs)
                    train_data.append(obs)
                    if player.player_name in winner_names:
                        labels.append(action_label)
                    else:
                        # replace action call/raise with fold
                        labels.append((ActionType.FOLD.value, -1))
            obs, _, done, _ = wrapped_env.step(action_label)

            it += 1

        return train_data, labels
