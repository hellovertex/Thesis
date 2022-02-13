from typing import List, Tuple, Dict
import numpy as np
from thesis.core.parser import PokerEpisode, Action, ActionType, Blind
from thesis.core.encoder import Encoder
from PokerRL.game.games import NoLimitHoldem
from thesis.core.encoder import PlayerInfo, Positions6Max
from PokerRL.game.Poker import Poker

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


class RLStateEncoder(Encoder):
    Observations = List[List]
    Actions_Taken = List[Tuple[int, int]]

    def __init__(self, env_wrapper_cls=None, currency_symbol='$'):
        self.env_wrapper_cls = env_wrapper_cls
        self._wrapped_env = None
        self._currency_symbol = currency_symbol

    @staticmethod
    def str_cards_to_list(cards: str):
        """See example below """
        # '[6h Ts Td 9c Jc]'
        rm_brackets = cards.replace('[', '').replace(']', '')
        # '6h Ts Td 9c Jc'
        card_list = rm_brackets.split(' ')
        # ['6h', 'Ts', 'Td', '9c', 'Jc']
        return card_list

    @staticmethod
    def build_action(action: Action, multiply_by=100):
        """Under Construction."""
        return action.action_type.value, int(float(action.raise_amount) * multiply_by)

    def make_blinds(self, blinds: List[Blind], multiply_by: int = 1):
        """Under Construction."""
        sb = blinds[0]
        assert sb.type == 'small blind'
        bb = blinds[1]
        assert bb.type == 'big blind'
        return int(sb.amount.split(self._currency_symbol)[1]) * multiply_by, \
               int(bb.amount.split(self._currency_symbol)[1]) * multiply_by

    def make_board_cards(self, board_cards: str):
        """Return 5 cards that we can prepend to the card deck so that the board will be drawn.
      Args:
        board_cards: for example '[6h Ts Td 9c Jc]'
      Returns:
        representation of board_cards that is understood by rl_env
        Example:
    """
        # '[6h Ts Td 9c Jc]' to ['6h', 'Ts', 'Td', '9c', 'Jc']
        card_list = self.str_cards_to_list(board_cards)
        assert len(card_list) == 5

        return [[DICT_RANK[card[0]], DICT_SUITE[card[1]]] for card in card_list]

    def make_showdown_hands(self, table, showdown):
        """Under Construction. """
        # initialize default hands
        player_hands = [Poker.CARD_NOT_DEALT_TOKEN_2D for _ in range(len(table))]

        # overwrite known hands
        for seat in table:
            for final_player in showdown:
                if seat.player_name == final_player.name:
                    # '[6h Ts]' to ['6h', 'Ts']
                    showdown_cards = self.str_cards_to_list(final_player.cards)
                    # ['6h', 'Ts'] to [[5,3], [5,0]]
                    hand = [[DICT_RANK[card[0]], DICT_SUITE[card[1]]] for card in showdown_cards]
                    # overwrite [[-127,127],[-127,-127]] with [[5,3], [5,0]]
                    player_hands[int(seat.position_index)] = hand
        return player_hands

    @staticmethod
    def _roll_position_indices(num_players: int, btn_idx: int) -> np.ndarray:
        """ Roll position indices, such that each seat is assigned correct position.
        Args:
          btn_idx: seat index (not seat number) of seat that is currently the Button.
                    Seats can be ["Seat 1", "Seat3", "Seat 5"]. If "Seat 5" is the Button,
                    then btn_idx=2
          num_players: Number of players currently at the table (not max. players).
        Returns: Assignment of position indices to seat numbers.

        Example: btn_idx=1
            # ==> np.roll([0,1,2], btn_idx) returns [2,0,1]:
            # The first  seat has position index 2, which is BB
            # The second seat has position index 0, which is BTN
            # The third  seat has position index 1, which is SB
        """
        # np.roll([0,1,2,3], 1) returns [3,0,1,2]  <== Button is at index 1 now
        return np.roll(np.arange(num_players), btn_idx)

    def make_table(self, episode: PokerEpisode) -> Tuple[PlayerInfo]:
        """Under Construction."""
        # Roll position indices, such that each seat is assigned correct position
        rolled_position_indices = self._roll_position_indices(episode.num_players, episode.btn_idx)

        # init {'BTN': None, 'SB': None,..., 'CO': None}
        player_info: Dict[str, PlayerInfo] = dict.fromkeys(
            [pos.name for pos in Positions6Max][:episode.num_players])

        # build PlayerInfo for each player
        for seat_idx, seat in enumerate(episode.player_stacks):
            seat_number = int(seat.seat_display_name[5])
            player_name = seat.player_name
            stack_size = float(seat.stack[1:])
            position_index = rolled_position_indices[seat_idx]
            position = Positions6Max(position_index).name
            player_info[position] = PlayerInfo(seat_number,  # 2
                                               position_index,  # 0
                                               position,  # 'BTN'
                                               player_name,  # 'JoeSchmoe Billy'
                                               stack_size)

        # Seat indices such that button is first, regardless of seat number
        players_ordered_starting_with_button = [v for v in player_info.values()]
        return tuple(players_ordered_starting_with_button)

    def _build_cards_state_dict(self, table: Tuple[PlayerInfo], episode: PokerEpisode):
        """Under Construction."""
        board_cards = self.make_board_cards(episode.board_cards)
        # --- set deck ---
        # cards are drawn without ghost cards, so we simply replace the first 5 cards of the deck
        # with the board cards that we have parsed
        deck = np.empty(shape=(13 * 4, 2), dtype=np.int8)
        deck[:len(board_cards)] = board_cards
        # make hands: np.ndarray(shape=(n_players, 2, 2))
        player_hands = self.make_showdown_hands(table, episode.showdown_hands)
        board = np.full((5, 2), Poker.CARD_NOT_DEALT_TOKEN_1D, dtype=np.int8)
        return {'deck': {'deck_remaining': deck},  # np.ndarray(shape=(52-n_cards*num_players, 2))
                'board': board,  # np.ndarray(shape=(n_cards, 2))
                'hand': player_hands}

    def _init_wrapped_env(self, table: Tuple[PlayerInfo], multiply_by=100):
        """Initializes environment used to generate observations.
        Assumes Btn is at index 0."""
        # get starting stacks, starting with button at index 0
        stacks = [player.stack_size for player in table]
        starting_stack_sizes_list = [int(float(stack) * multiply_by) for stack in stacks]

        # make args for env
        args = NoLimitHoldem.ARGS_CLS(n_seats=len(table),
                                      starting_stack_sizes_list=starting_stack_sizes_list)
        # return wrapped env instance
        env = NoLimitHoldem(is_evaluating=True,
                            env_args=args,
                            lut_holder=NoLimitHoldem.get_lut_holder())
        self._wrapped_env = self.env_wrapper_cls(env)

    def _simulate_environment(self, env, episode, cards_state_dict, table):
        """Under Construction."""
        showdown_players = [player.name for player in episode.showdown_hands]
        state_dict = {'deck': cards_state_dict, 'table': table}
        obs, _, done, _ = env.reset(state_dict=state_dict)

        # --- Step Environment with action --- #
        observations = []
        actions = []
        it = 0
        while not done:
            action = episode.actions_total['as_sequence'][it]
            action_formatted = self.build_action(action)
            # store up to two actions per player per stage
            # self._actions_per_stage[action.player_name][action.stage].append(action_formatted)
            next_to_act = env.current_player.seat_id
            for player in table:
                # if player reached showdown (we can see his cards)
                if player.position_index == next_to_act and player.player_name in showdown_players:
                    observations.append(obs)
                    # player that won showdown -- can be multiple (split pot)
                    if player.player_name in [winner.name for winner in episode.winners]:
                        action_label = self._wrapped_env.discretize(action_formatted)
                        # actions.append(action_formatted)  # use his action as supervised label
                    # player that lost showdown
                    else:
                        # replace action call/raise with fold
                        action_label = self._wrapped_env.discretize((ActionType.FOLD.value, -1))
                        # actions.append((ActionType.FOLD.value, -1))  # replace action with FOLD for now
                    actions.append(action_label)
            obs, _, done, _ = env.step(action_formatted)
            it += 1
        return observations, actions

    def encode_episode(self, episode: PokerEpisode) -> Tuple[Observations, Actions_Taken]:
        """Runs environment with steps from PokerEpisode.
        Returns observations and corresponding actions of players that made it to showdown."""
        # utils
        table = self.make_table(episode)

        # Initialize environment for simulation of PokerEpisode
        # todo: pass env_cls as argument (N_BOARD_CARDS etc. gets accessible)
        self._init_wrapped_env(table)
        self._wrapped_env.SMALL_BLIND, self._wrapped_env.BIG_BLIND = self.make_blinds(episode.blinds, multiply_by=100)
        cards_state_dict = self._build_cards_state_dict(table, episode)

        # Collect observations and actions, observations are possibly augmented
        return self._simulate_environment(env=self._wrapped_env,
                                          episode=episode,
                                          cards_state_dict=cards_state_dict,
                                          table=table)
