import enum
import numpy as np
from gym import spaces
from collections import defaultdict, deque
from PokerRL.game.Poker import Poker

from thesis.baselines.supervised_learning.data.core.encoder import Positions6Max
from thesis.baselines.supervised_learning.data.core.wrapper import WrapperPokerRL
from thesis.baselines.supervised_learning.data.canonical_vectorizer import CanonicalVectorizer


class ActionSpace(enum.IntEnum):
    """Under Construction"""
    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISE_MIN_OR_3BB = 3
    RAISE_HALF_POT = 4
    RAISE_POT = 5
    ALL_IN = 6
    SMALL_BLIND = 7
    BIG_BLIND = 8


class ActionHistory:
    # noinspection PyTypeChecker
    def __init__(self, max_players, max_actions_per_player_per_stage):
        self._max_players = max_players
        self._max_actions_per_player_per_stage = max_actions_per_player_per_stage
        self.deque = {}

        for pos in range(self._max_players):
            # create default dictionary for current player for each stage
            # default dictionary stores only the last two actions per stage per player
            self.deque[Positions6Max(pos).value] = defaultdict(
                lambda: deque(maxlen=max_actions_per_player_per_stage),
                keys=['preflop', 'flop', 'turn', 'river'])

    def __str__(self):
        representation = ""
        for player_seat_id in range(self._max_players):
            representation += f'--- Player {player_seat_id}:---\n' + \
                              f'Actions preflop: {self.deque[player_seat_id]["preflop"]}\n' + \
                              f'Actions flop: {self.deque[player_seat_id]["flop"]}\n' + \
                              f'Actions turn: {self.deque[player_seat_id]["turn"]}\n' + \
                              f'Actions river: {self.deque[player_seat_id]["river"]}\n'
        return representation

    def __repr__(self):
        return self.__str__()


class ActionHistoryWrapper(WrapperPokerRL):

    def __init__(self, env):
        """
        Args:
            env (PokerEnv subclass instance):   The environment instance to be wrapped
        """
        super().__init__(env=env)
        self._player_hands = []
        self._rounds = ['preflop', 'flop', 'turn', 'river']
        self._actions_per_stage = ActionHistory(max_players=6, max_actions_per_player_per_stage=2)
        self._player_who_acted = None
        # experimental
        self._actions_per_stage_discretized = ActionHistory(max_players=6, max_actions_per_player_per_stage=2)

    # _______________________________ Overridden ________________________________
    def _before_step(self, action):
        """
        Steps the environment from an action of the natural action representation to the environment.

        Returns:
            obs, reward, done, info
        """
        # store action in history buffer
        self._pushback_action(action,
                              player_who_acted=self.env.current_player.seat_id,
                              in_which_stage=self.env.current_round)
        self._player_who_acted = self.env.current_player.seat_id

    def _before_reset(self, config=None):
        # for the initial case of the environment reset, we manually put player index to 0
        # so that observation will be rolled relative to self
        if config is not None:
            self._player_hands = config['deck_state_dict']['hand']
        self._player_who_acted = 0


    # _______________________________ Action History ________________________________

    def discretize(self, action_formatted):
        if action_formatted[0] == 2:  # action is raise
            pot_size = self.env.get_all_winnable_money()
            raise_amt = action_formatted[1]
            if raise_amt < pot_size / 2:
                return ActionSpace.RAISE_MIN_OR_3BB
            elif raise_amt < pot_size:
                return ActionSpace.RAISE_HALF_POT
            elif raise_amt < 2 * pot_size:
                return ActionSpace.RAISE_POT
            else:
                return ActionSpace.ALL_IN
        else:  # action is fold or check/call
            action_discretized = action_formatted[0]
        return action_discretized

    def _pushback_action(self, action_formatted, player_who_acted, in_which_stage):
        # part of observation
        self._actions_per_stage.deque[player_who_acted][
            self._rounds[in_which_stage]].append(action_formatted)

        action_discretized = self.discretize(action_formatted)
        # for the neural network labels
        self._actions_per_stage_discretized.deque[player_who_acted][
            self._rounds[in_which_stage]].append(action_discretized)

    # _______________________________ Override to Augment observation ________________________________
    def get_current_obs(self, env_obs):
        """Implement this to encode Action History into observation"""
        raise NotImplementedError



# noinspection DuplicatedCode
class AugmentObservationWrapper(ActionHistoryWrapper):

    def __init__(self, env):
        super().__init__(env=env)
        # todo: (?) check how obs is normalized to avoid small floats
        self._normalization_sum = float(
            sum([s.starting_stack_this_episode for s in self.env.seats])
        ) / self.env.N_SEATS
        self.num_players = env.N_SEATS
        self.max_players = 6
        self.num_board_cards = 5
        self.observation_space, self.obs_idx_dict, self.obs_parts_idxs_dict = self._construct_obs_space()
        self._vectorizer = CanonicalVectorizer(num_players=self.num_players,
                                               obs_idx_dict=self.env.obs_idx_dict,
                                               # btn pos used to return obs relative to self
                                               btn_pos=self.env.BTN_POS)

    def get_current_obs(self, env_obs):
        obs = self._vectorizer.vectorize(env_obs, self._player_who_acted, action_history=self._actions_per_stage,
                                         player_hands=self._player_hands, normalization=self._normalization_sum)
        # self.print_augmented_obs(obs)
        return obs

    def _construct_obs_space(self):
        """
        The maximum all chip-values can reach is n_seats, because we normalize by dividing by the average starting stack
        """
        obs_idx_dict = {}
        obs_parts_idxs_dict = {
            "board": [],
            "players": [[] for _ in range(self.max_players)],
            "table_state": [],
            "player_cards": [[] for _ in range(self.max_players)],
            "action_history": [[] for _ in range(len(self._rounds))]
        }
        next_idx = [0]  # list is a mutatable object. int not.

        def get_discrete(size, name, _curr_idx):
            obs_idx_dict[name] = _curr_idx[0]
            _curr_idx[0] += 1
            return spaces.Discrete(size)

        def get_new_box(name, _curr_idx, high, low=0):
            obs_idx_dict[name] = _curr_idx[0]
            _curr_idx[0] += 1
            return spaces.Box(low=low, high=high, shape=(1,), dtype=np.float32)

        # __________________________  Public Information About Game State  _________________________
        _k = next_idx[0]
        _table_space = [  # (blinds are in obs to give the agent a perspective on starting stack after normalization
            get_new_box("ante", next_idx, self.max_players),  # .................................... self.ANTE
            get_new_box("small_blind", next_idx, self.max_players),  # ............................. self.SMALL_BLIND
            get_new_box("big_blind", next_idx, self.max_players),  # ............................... self.BIG_BLIND
            get_new_box("min_raise", next_idx, self.max_players),  # ............................... min total raise
            get_new_box("pot_amt", next_idx, self.max_players),  # ................................. main_pot amount
            get_new_box("total_to_call", next_idx, self.max_players),  # ........................... total_to_call
            # get_new_box("last_action_how_much", next_idx, self.max_players),  # .................... self.last_action[1]
        ]
        # for i in range(3):  # .................................................................. self.last_action[0]
        #   _table_space.append(get_discrete(1, "last_action_what_" + str(i), next_idx))
        #
        # for i in range(self.max_players):  # ....................................................... self.last_action[2]
        #   _table_space.append(get_discrete(1, "last_action_who_" + str(i), next_idx))

        for i in range(self.max_players):  # ....................................................... curr_player.seat_id
            _table_space.append(get_discrete(1, "p" + str(i) + "_acts_next", next_idx))

        for i in range(len(self._rounds)):  # ...................................... round onehot
            _table_space.append(get_discrete(1, "round_" + Poker.INT2STRING_ROUND[i], next_idx)),

        for i in range(self.max_players):  # ....................................................... side pots
            _table_space.append(get_new_box("side_pot_" + str(i), next_idx, 1))

        # add to parts_dict for possible slicing for agents.
        obs_parts_idxs_dict["table_state"] += list(range(_k, next_idx[0]))

        # __________________________  Public Information About Each Player  ________________________
        _player_space = []
        for i in range(self.max_players):
            _k = next_idx[0]
            _player_space += [
                get_new_box("stack_p" + str(i), next_idx, self.max_players),  # ..................... stack
                get_new_box("curr_bet_p" + str(i), next_idx, self.max_players),  # .................. current_bet
                get_discrete(1, "has_folded_this_episode_p" + str(i), next_idx),  # ............. folded_this_epis
                get_discrete(1, "is_allin_p" + str(i), next_idx),  # ............................ is_allin
            ]
            for j in range(self.max_players):
                _player_space.append(
                    get_discrete(1, "side_pot_rank_p" + str(i) + "_is_" + str(j), next_idx))  # . side_pot_rank

            # add to parts_dict for possible slicing for agents
            obs_parts_idxs_dict["players"][i] += list(range(_k, next_idx[0]))

        # _______________________________  Public cards (i.e. board)  ______________________________
        _board_space = []
        _k = next_idx[0]
        for i in range(self.num_board_cards):

            x = []
            for j in range(self.env.N_RANKS):  # .................................................... rank
                x.append(get_discrete(1, str(i) + "th_board_card_rank_" + str(j), next_idx))

            for j in range(self.env.N_SUITS):  # .................................................... suit
                x.append(get_discrete(1, str(i) + "th_board_card_suit_" + str(j), next_idx))

            _board_space += x

        # add to parts_dict for possible slicing for agents
        obs_parts_idxs_dict["board"] += list(range(_k, next_idx[0]))

        # _______________________________  Private Cards (i.e. players hands)  ______________________________
        # add to parts_dict for possible slicing for agents
        _handcards_space = []

        for i in range(self.max_players):
            _k = next_idx[0]
            for k in range(self.env.N_HOLE_CARDS):
                x = []
                for j in range(self.env.N_RANKS):  # .................................................... rank
                    x.append(get_discrete(1, str(i) + f"th_player_card_{k}_rank_" + str(j), next_idx))

                for j in range(self.env.N_SUITS):  # .................................................... suit
                    x.append(get_discrete(1, str(i) + f"th_player_card_{k}_suit_" + str(j), next_idx))

                _handcards_space += x

            obs_parts_idxs_dict["player_cards"][i] += list(range(_k, next_idx[0]))
        # _______________________________  Action History (max 2 /stage/player)  ______________________________
        # add to parts_dict for possible slicing for agents
        # preflop_player_0_action_0_how_much
        # preflop_player_0_action_0_what_0
        # preflop_player_0_action_0_what_1
        # preflop_player_0_action_0_what_2
        # preflop_player_0_action_1_how_much
        # preflop_player_0_action_1_what_0
        # preflop_player_0_action_1_what_1
        # preflop_player_0_action_1_what_2
        _action_history_space = []

        for i in range(len(self._rounds)):
            _k = next_idx[0]
            for j in range(self.max_players):
                for a in [0, 1]:
                    _action_history_space.append(
                        get_new_box(f"{self._rounds[i]}_player_{j}_action_{a}_how_much", next_idx, self.max_players))
                    for k in range(3):
                        _action_history_space.append(
                            get_discrete(1, f"{self._rounds[i]}_player_{j}_action_{a}_what_{k}", next_idx)
                        )
            obs_parts_idxs_dict["action_history"][i] += list(range(_k, next_idx[0]))
        # preflop, flop, turn, river : [action0, action1], []

        # __________________________  Return Complete _Observation Space  __________________________
        # Tuple (lots of spaces.Discrete and spaces.Box)
        _observation_space = spaces.Tuple(_table_space + _player_space + _board_space)
        _observation_space.shape = [len(_observation_space.spaces)]
        return _observation_space, obs_idx_dict, obs_parts_idxs_dict

    def print_augmented_obs(self, obs):
        """Can be used for debugging. Printed Observation example:
        ______________________________________ Printing _Observation _________________________________________
                              ante:   0.0
                       small_blind:   0.0032992411870509386
                         big_blind:   0.006598482374101877
                         min_raise:   0.013196964748203754
                           pot_amt:   0.0
                     total_to_call:   0.006598482374101877
                      p0_acts_next:   1.0
                      p1_acts_next:   0.0
                      p2_acts_next:   0.0
                      p3_acts_next:   1.0
                      p4_acts_next:   0.0
                      p5_acts_next:   0.0
                     round_preflop:   1.0
                        round_flop:   0.0
                        round_turn:   0.0
                       round_river:   0.0
                        side_pot_0:   0.0
                        side_pot_1:   0.0
                        side_pot_2:   0.0
                        side_pot_3:   0.0
                        side_pot_4:   0.0
                        side_pot_5:   0.0
                          stack_p0:   1.106103539466858
                       curr_bet_p0:   0.0
        has_folded_this_episode_p0:   0.0
                       is_allin_p0:   0.0
             side_pot_rank_p0_is_0:   0.0
             side_pot_rank_p0_is_1:   0.0
             side_pot_rank_p0_is_2:   0.0
             side_pot_rank_p0_is_3:   0.0
             side_pot_rank_p0_is_4:   0.0
             side_pot_rank_p0_is_5:   0.0
                          stack_p1:   0.5709006786346436
                       curr_bet_p1:   0.0032992411870509386
        has_folded_this_episode_p1:   0.0
                       is_allin_p1:   0.0
             side_pot_rank_p1_is_0:   0.0
             side_pot_rank_p1_is_1:   0.0
             side_pot_rank_p1_is_2:   0.0
             side_pot_rank_p1_is_3:   0.0
             side_pot_rank_p1_is_4:   0.0
             side_pot_rank_p1_is_5:   0.0
                          stack_p2:   1.3130979537963867
                       curr_bet_p2:   0.006598482374101877
        has_folded_this_episode_p2:   0.0
                       is_allin_p2:   0.0
             side_pot_rank_p2_is_0:   0.0
             side_pot_rank_p2_is_1:   0.0
             side_pot_rank_p2_is_2:   0.0
             side_pot_rank_p2_is_3:   0.0
             side_pot_rank_p2_is_4:   0.0
             side_pot_rank_p2_is_5:   0.0
                          stack_p3:   1.106103539466858
                       curr_bet_p3:   0.0
        has_folded_this_episode_p3:   0.0
                       is_allin_p3:   0.0
             side_pot_rank_p3_is_0:   0.0
             side_pot_rank_p3_is_1:   0.0
             side_pot_rank_p3_is_2:   0.0
             side_pot_rank_p3_is_3:   0.0
             side_pot_rank_p3_is_4:   0.0
             side_pot_rank_p3_is_5:   0.0
                          stack_p4:   0.5709006786346436
                       curr_bet_p4:   0.0032992411870509386
        has_folded_this_episode_p4:   0.0
                       is_allin_p4:   0.0
             side_pot_rank_p4_is_0:   0.0
             side_pot_rank_p4_is_1:   0.0
             side_pot_rank_p4_is_2:   0.0
             side_pot_rank_p4_is_3:   0.0
             side_pot_rank_p4_is_4:   0.0
             side_pot_rank_p4_is_5:   0.0
                          stack_p5:   1.3130979537963867
                       curr_bet_p5:   0.006598482374101877
        has_folded_this_episode_p5:   0.0
                       is_allin_p5:   0.0
             side_pot_rank_p5_is_0:   0.0
             side_pot_rank_p5_is_1:   0.0
             side_pot_rank_p5_is_2:   0.0
             side_pot_rank_p5_is_3:   0.0
             side_pot_rank_p5_is_4:   0.0
             side_pot_rank_p5_is_5:   0.0
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
          0th_player_card_0_rank_0:   0.0
          0th_player_card_0_rank_1:   0.0
          0th_player_card_0_rank_2:   0.0
          0th_player_card_0_rank_3:   0.0
          0th_player_card_0_rank_4:   0.0
          0th_player_card_0_rank_5:   1.0
          0th_player_card_0_rank_6:   0.0
          0th_player_card_0_rank_7:   0.0
          0th_player_card_0_rank_8:   0.0
          0th_player_card_0_rank_9:   0.0
         0th_player_card_0_rank_10:   0.0
         0th_player_card_0_rank_11:   0.0
         0th_player_card_0_rank_12:   0.0
           0th_board_card_0_suit_0:   0.0
           0th_board_card_0_suit_1:   0.0
           0th_board_card_0_suit_2:   0.0
           0th_board_card_0_suit_3:   1.0
          0th_player_card_1_rank_0:   0.0
          0th_player_card_1_rank_1:   0.0
          0th_player_card_1_rank_2:   0.0
          0th_player_card_1_rank_3:   0.0
          0th_player_card_1_rank_4:   0.0
          0th_player_card_1_rank_5:   1.0
          0th_player_card_1_rank_6:   0.0
          0th_player_card_1_rank_7:   0.0
          0th_player_card_1_rank_8:   0.0
          0th_player_card_1_rank_9:   0.0
         0th_player_card_1_rank_10:   0.0
         0th_player_card_1_rank_11:   0.0
         0th_player_card_1_rank_12:   0.0
           0th_board_card_1_suit_0:   1.0
           0th_board_card_1_suit_1:   0.0
           0th_board_card_1_suit_2:   0.0
           0th_board_card_1_suit_3:   0.0
          1th_player_card_0_rank_0:   0.0
          1th_player_card_0_rank_1:   0.0
          1th_player_card_0_rank_2:   0.0
          1th_player_card_0_rank_3:   0.0
          1th_player_card_0_rank_4:   0.0
          1th_player_card_0_rank_5:   0.0
          1th_player_card_0_rank_6:   0.0
          1th_player_card_0_rank_7:   0.0
          1th_player_card_0_rank_8:   0.0
          1th_player_card_0_rank_9:   0.0
         1th_player_card_0_rank_10:   0.0
         1th_player_card_0_rank_11:   0.0
         1th_player_card_0_rank_12:   1.0
           1th_board_card_0_suit_0:   1.0
           1th_board_card_0_suit_1:   0.0
           1th_board_card_0_suit_2:   0.0
           1th_board_card_0_suit_3:   0.0
          1th_player_card_1_rank_0:   0.0
          1th_player_card_1_rank_1:   0.0
          1th_player_card_1_rank_2:   0.0
          1th_player_card_1_rank_3:   0.0
          1th_player_card_1_rank_4:   0.0
          1th_player_card_1_rank_5:   0.0
          1th_player_card_1_rank_6:   0.0
          1th_player_card_1_rank_7:   0.0
          1th_player_card_1_rank_8:   0.0
          1th_player_card_1_rank_9:   1.0
         1th_player_card_1_rank_10:   0.0
         1th_player_card_1_rank_11:   0.0
         1th_player_card_1_rank_12:   0.0
           1th_board_card_1_suit_0:   0.0
           1th_board_card_1_suit_1:   1.0
           1th_board_card_1_suit_2:   0.0
           1th_board_card_1_suit_3:   0.0
          2th_player_card_0_rank_0:   1.0
          2th_player_card_0_rank_1:   0.0
          2th_player_card_0_rank_2:   0.0
          2th_player_card_0_rank_3:   0.0
          2th_player_card_0_rank_4:   0.0
          2th_player_card_0_rank_5:   0.0
          2th_player_card_0_rank_6:   0.0
          2th_player_card_0_rank_7:   0.0
          2th_player_card_0_rank_8:   0.0
          2th_player_card_0_rank_9:   0.0
         2th_player_card_0_rank_10:   0.0
         2th_player_card_0_rank_11:   0.0
         2th_player_card_0_rank_12:   0.0
           2th_board_card_0_suit_0:   1.0
           2th_board_card_0_suit_1:   0.0
           2th_board_card_0_suit_2:   0.0
           2th_board_card_0_suit_3:   0.0
          2th_player_card_1_rank_0:   1.0
          2th_player_card_1_rank_1:   0.0
          2th_player_card_1_rank_2:   0.0
          2th_player_card_1_rank_3:   0.0
          2th_player_card_1_rank_4:   0.0
          2th_player_card_1_rank_5:   0.0
          2th_player_card_1_rank_6:   0.0
          2th_player_card_1_rank_7:   0.0
          2th_player_card_1_rank_8:   0.0
          2th_player_card_1_rank_9:   0.0
         2th_player_card_1_rank_10:   0.0
         2th_player_card_1_rank_11:   0.0
         2th_player_card_1_rank_12:   0.0
           2th_board_card_1_suit_0:   1.0
           2th_board_card_1_suit_1:   0.0
           2th_board_card_1_suit_2:   0.0
           2th_board_card_1_suit_3:   0.0
          3th_player_card_0_rank_0:   0.0
          3th_player_card_0_rank_1:   0.0
          3th_player_card_0_rank_2:   0.0
          3th_player_card_0_rank_3:   0.0
          3th_player_card_0_rank_4:   0.0
          3th_player_card_0_rank_5:   1.0
          3th_player_card_0_rank_6:   0.0
          3th_player_card_0_rank_7:   0.0
          3th_player_card_0_rank_8:   0.0
          3th_player_card_0_rank_9:   0.0
         3th_player_card_0_rank_10:   0.0
         3th_player_card_0_rank_11:   0.0
         3th_player_card_0_rank_12:   0.0
           3th_board_card_0_suit_0:   0.0
           3th_board_card_0_suit_1:   0.0
           3th_board_card_0_suit_2:   0.0
           3th_board_card_0_suit_3:   1.0
          3th_player_card_1_rank_0:   0.0
          3th_player_card_1_rank_1:   0.0
          3th_player_card_1_rank_2:   0.0
          3th_player_card_1_rank_3:   0.0
          3th_player_card_1_rank_4:   0.0
          3th_player_card_1_rank_5:   1.0
          3th_player_card_1_rank_6:   0.0
          3th_player_card_1_rank_7:   0.0
          3th_player_card_1_rank_8:   0.0
          3th_player_card_1_rank_9:   0.0
         3th_player_card_1_rank_10:   0.0
         3th_player_card_1_rank_11:   0.0
         3th_player_card_1_rank_12:   0.0
           3th_board_card_1_suit_0:   1.0
           3th_board_card_1_suit_1:   0.0
           3th_board_card_1_suit_2:   0.0
           3th_board_card_1_suit_3:   0.0
          4th_player_card_0_rank_0:   0.0
          4th_player_card_0_rank_1:   0.0
          4th_player_card_0_rank_2:   0.0
          4th_player_card_0_rank_3:   0.0
          4th_player_card_0_rank_4:   0.0
          4th_player_card_0_rank_5:   0.0
          4th_player_card_0_rank_6:   0.0
          4th_player_card_0_rank_7:   0.0
          4th_player_card_0_rank_8:   0.0
          4th_player_card_0_rank_9:   0.0
         4th_player_card_0_rank_10:   0.0
         4th_player_card_0_rank_11:   0.0
         4th_player_card_0_rank_12:   1.0
           4th_board_card_0_suit_0:   1.0
           4th_board_card_0_suit_1:   0.0
           4th_board_card_0_suit_2:   0.0
           4th_board_card_0_suit_3:   0.0
          4th_player_card_1_rank_0:   0.0
          4th_player_card_1_rank_1:   0.0
          4th_player_card_1_rank_2:   0.0
          4th_player_card_1_rank_3:   0.0
          4th_player_card_1_rank_4:   0.0
          4th_player_card_1_rank_5:   0.0
          4th_player_card_1_rank_6:   0.0
          4th_player_card_1_rank_7:   0.0
          4th_player_card_1_rank_8:   0.0
          4th_player_card_1_rank_9:   1.0
         4th_player_card_1_rank_10:   0.0
         4th_player_card_1_rank_11:   0.0
         4th_player_card_1_rank_12:   0.0
           4th_board_card_1_suit_0:   0.0
           4th_board_card_1_suit_1:   1.0
           4th_board_card_1_suit_2:   0.0
           4th_board_card_1_suit_3:   0.0
          5th_player_card_0_rank_0:   1.0
          5th_player_card_0_rank_1:   0.0
          5th_player_card_0_rank_2:   0.0
          5th_player_card_0_rank_3:   0.0
          5th_player_card_0_rank_4:   0.0
          5th_player_card_0_rank_5:   0.0
          5th_player_card_0_rank_6:   0.0
          5th_player_card_0_rank_7:   0.0
          5th_player_card_0_rank_8:   0.0
          5th_player_card_0_rank_9:   0.0
         5th_player_card_0_rank_10:   0.0
         5th_player_card_0_rank_11:   0.0
         5th_player_card_0_rank_12:   0.0
           5th_board_card_0_suit_0:   1.0
           5th_board_card_0_suit_1:   0.0
           5th_board_card_0_suit_2:   0.0
           5th_board_card_0_suit_3:   0.0
          5th_player_card_1_rank_0:   1.0
          5th_player_card_1_rank_1:   0.0
          5th_player_card_1_rank_2:   0.0
          5th_player_card_1_rank_3:   0.0
          5th_player_card_1_rank_4:   0.0
          5th_player_card_1_rank_5:   0.0
          5th_player_card_1_rank_6:   0.0
          5th_player_card_1_rank_7:   0.0
          5th_player_card_1_rank_8:   0.0
          5th_player_card_1_rank_9:   0.0
         5th_player_card_1_rank_10:   0.0
         5th_player_card_1_rank_11:   0.0
         5th_player_card_1_rank_12:   0.0
           5th_board_card_1_suit_0:   1.0
           5th_board_card_1_suit_1:   0.0
           5th_board_card_1_suit_2:   0.0
           5th_board_card_1_suit_3:   0.0
preflop_player_0_action_0_how_much:   0.0
  preflop_player_0_action_0_what_0:   0.0
  preflop_player_0_action_0_what_1:   0.0
  preflop_player_0_action_0_what_2:   0.0
preflop_player_0_action_1_how_much:   0.0
  preflop_player_0_action_1_what_0:   0.0
  preflop_player_0_action_1_what_1:   0.0
  preflop_player_0_action_1_what_2:   0.0
preflop_player_1_action_0_how_much:   0.0
  preflop_player_1_action_0_what_0:   0.0
  preflop_player_1_action_0_what_1:   0.0
  preflop_player_1_action_0_what_2:   0.0
preflop_player_1_action_1_how_much:   0.0
  preflop_player_1_action_1_what_0:   0.0
  preflop_player_1_action_1_what_1:   0.0
  preflop_player_1_action_1_what_2:   0.0
preflop_player_2_action_0_how_much:   0.0
  preflop_player_2_action_0_what_0:   0.0
  preflop_player_2_action_0_what_1:   0.0
  preflop_player_2_action_0_what_2:   0.0
preflop_player_2_action_1_how_much:   0.0
  preflop_player_2_action_1_what_0:   0.0
  preflop_player_2_action_1_what_1:   0.0
  preflop_player_2_action_1_what_2:   0.0
preflop_player_3_action_0_how_much:   0.0
  preflop_player_3_action_0_what_0:   0.0
  preflop_player_3_action_0_what_1:   0.0
  preflop_player_3_action_0_what_2:   0.0
preflop_player_3_action_1_how_much:   0.0
  preflop_player_3_action_1_what_0:   0.0
  preflop_player_3_action_1_what_1:   0.0
  preflop_player_3_action_1_what_2:   0.0
preflop_player_4_action_0_how_much:   0.0
  preflop_player_4_action_0_what_0:   0.0
  preflop_player_4_action_0_what_1:   0.0
  preflop_player_4_action_0_what_2:   0.0
preflop_player_4_action_1_how_much:   0.0
  preflop_player_4_action_1_what_0:   0.0
  preflop_player_4_action_1_what_1:   0.0
  preflop_player_4_action_1_what_2:   0.0
preflop_player_5_action_0_how_much:   0.0
  preflop_player_5_action_0_what_0:   0.0
  preflop_player_5_action_0_what_1:   0.0
  preflop_player_5_action_0_what_2:   0.0
preflop_player_5_action_1_how_much:   0.0
  preflop_player_5_action_1_what_0:   0.0
  preflop_player_5_action_1_what_1:   0.0
  preflop_player_5_action_1_what_2:   0.0
   flop_player_0_action_0_how_much:   0.0
     flop_player_0_action_0_what_0:   0.0
     flop_player_0_action_0_what_1:   0.0
     flop_player_0_action_0_what_2:   0.0
   flop_player_0_action_1_how_much:   0.0
     flop_player_0_action_1_what_0:   0.0
     flop_player_0_action_1_what_1:   0.0
     flop_player_0_action_1_what_2:   0.0
   flop_player_1_action_0_how_much:   0.0
     flop_player_1_action_0_what_0:   0.0
     flop_player_1_action_0_what_1:   0.0
     flop_player_1_action_0_what_2:   0.0
   flop_player_1_action_1_how_much:   0.0
     flop_player_1_action_1_what_0:   0.0
     flop_player_1_action_1_what_1:   0.0
     flop_player_1_action_1_what_2:   0.0
   flop_player_2_action_0_how_much:   0.0
     flop_player_2_action_0_what_0:   0.0
     flop_player_2_action_0_what_1:   0.0
     flop_player_2_action_0_what_2:   0.0
   flop_player_2_action_1_how_much:   0.0
     flop_player_2_action_1_what_0:   0.0
     flop_player_2_action_1_what_1:   0.0
     flop_player_2_action_1_what_2:   0.0
   flop_player_3_action_0_how_much:   0.0
     flop_player_3_action_0_what_0:   0.0
     flop_player_3_action_0_what_1:   0.0
     flop_player_3_action_0_what_2:   0.0
   flop_player_3_action_1_how_much:   0.0
     flop_player_3_action_1_what_0:   0.0
     flop_player_3_action_1_what_1:   0.0
     flop_player_3_action_1_what_2:   0.0
   flop_player_4_action_0_how_much:   0.0
     flop_player_4_action_0_what_0:   0.0
     flop_player_4_action_0_what_1:   0.0
     flop_player_4_action_0_what_2:   0.0
   flop_player_4_action_1_how_much:   0.0
     flop_player_4_action_1_what_0:   0.0
     flop_player_4_action_1_what_1:   0.0
     flop_player_4_action_1_what_2:   0.0
   flop_player_5_action_0_how_much:   0.0
     flop_player_5_action_0_what_0:   0.0
     flop_player_5_action_0_what_1:   0.0
     flop_player_5_action_0_what_2:   0.0
   flop_player_5_action_1_how_much:   0.0
     flop_player_5_action_1_what_0:   0.0
     flop_player_5_action_1_what_1:   0.0
     flop_player_5_action_1_what_2:   0.0
   turn_player_0_action_0_how_much:   0.0
     turn_player_0_action_0_what_0:   0.0
     turn_player_0_action_0_what_1:   0.0
     turn_player_0_action_0_what_2:   0.0
   turn_player_0_action_1_how_much:   0.0
     turn_player_0_action_1_what_0:   0.0
     turn_player_0_action_1_what_1:   0.0
     turn_player_0_action_1_what_2:   0.0
   turn_player_1_action_0_how_much:   0.0
     turn_player_1_action_0_what_0:   0.0
     turn_player_1_action_0_what_1:   0.0
     turn_player_1_action_0_what_2:   0.0
   turn_player_1_action_1_how_much:   0.0
     turn_player_1_action_1_what_0:   0.0
     turn_player_1_action_1_what_1:   0.0
     turn_player_1_action_1_what_2:   0.0
   turn_player_2_action_0_how_much:   0.0
     turn_player_2_action_0_what_0:   0.0
     turn_player_2_action_0_what_1:   0.0
     turn_player_2_action_0_what_2:   0.0
   turn_player_2_action_1_how_much:   0.0
     turn_player_2_action_1_what_0:   0.0
     turn_player_2_action_1_what_1:   0.0
     turn_player_2_action_1_what_2:   0.0
   turn_player_3_action_0_how_much:   0.0
     turn_player_3_action_0_what_0:   0.0
     turn_player_3_action_0_what_1:   0.0
     turn_player_3_action_0_what_2:   0.0
   turn_player_3_action_1_how_much:   0.0
     turn_player_3_action_1_what_0:   0.0
     turn_player_3_action_1_what_1:   0.0
     turn_player_3_action_1_what_2:   0.0
   turn_player_4_action_0_how_much:   0.0
     turn_player_4_action_0_what_0:   0.0
     turn_player_4_action_0_what_1:   0.0
     turn_player_4_action_0_what_2:   0.0
   turn_player_4_action_1_how_much:   0.0
     turn_player_4_action_1_what_0:   0.0
     turn_player_4_action_1_what_1:   0.0
     turn_player_4_action_1_what_2:   0.0
   turn_player_5_action_0_how_much:   0.0
     turn_player_5_action_0_what_0:   0.0
     turn_player_5_action_0_what_1:   0.0
     turn_player_5_action_0_what_2:   0.0
   turn_player_5_action_1_how_much:   0.0
     turn_player_5_action_1_what_0:   0.0
     turn_player_5_action_1_what_1:   0.0
     turn_player_5_action_1_what_2:   0.0
  river_player_0_action_0_how_much:   0.0
    river_player_0_action_0_what_0:   0.0
    river_player_0_action_0_what_1:   0.0
    river_player_0_action_0_what_2:   0.0
  river_player_0_action_1_how_much:   0.0
    river_player_0_action_1_what_0:   0.0
    river_player_0_action_1_what_1:   0.0
    river_player_0_action_1_what_2:   0.0
  river_player_1_action_0_how_much:   0.0
    river_player_1_action_0_what_0:   0.0
    river_player_1_action_0_what_1:   0.0
    river_player_1_action_0_what_2:   0.0
  river_player_1_action_1_how_much:   0.0
    river_player_1_action_1_what_0:   0.0
    river_player_1_action_1_what_1:   0.0
    river_player_1_action_1_what_2:   0.0
  river_player_2_action_0_how_much:   0.0
    river_player_2_action_0_what_0:   0.0
    river_player_2_action_0_what_1:   0.0
    river_player_2_action_0_what_2:   0.0
  river_player_2_action_1_how_much:   0.0
    river_player_2_action_1_what_0:   0.0
    river_player_2_action_1_what_1:   0.0
    river_player_2_action_1_what_2:   0.0
  river_player_3_action_0_how_much:   0.0
    river_player_3_action_0_what_0:   0.0
    river_player_3_action_0_what_1:   0.0
    river_player_3_action_0_what_2:   0.0
  river_player_3_action_1_how_much:   0.0
    river_player_3_action_1_what_0:   0.0
    river_player_3_action_1_what_1:   0.0
    river_player_3_action_1_what_2:   0.0
  river_player_4_action_0_how_much:   0.0
    river_player_4_action_0_what_0:   0.0
    river_player_4_action_0_what_1:   0.0
    river_player_4_action_0_what_2:   0.0
  river_player_4_action_1_how_much:   0.0
    river_player_4_action_1_what_0:   0.0
    river_player_4_action_1_what_1:   0.0
    river_player_4_action_1_what_2:   0.0
  river_player_5_action_0_how_much:   0.0
    river_player_5_action_0_what_0:   0.0
    river_player_5_action_0_what_1:   0.0
    river_player_5_action_0_what_2:   0.0
  river_player_5_action_1_how_much:   0.0
    river_player_5_action_1_what_0:   0.0
    river_player_5_action_1_what_1:   0.0
    river_player_5_action_1_what_2:   0.0"""
        print("______________________________________ Printing _Observation _________________________________________")
        names = [e + ":  " for e in list(self.obs_idx_dict.keys())]
        str_len = max([len(e) for e in names])
        for name, key in zip(names, list(self.obs_idx_dict.keys())):
            name = name.rjust(str_len)
            print(name, obs[self.obs_idx_dict[key]])

    @property
    def current_player(self):
        return self.env.current_player
