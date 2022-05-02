from typing import Optional

from pydantic import BaseModel


class Card(BaseModel):
    name: str
    suit: int
    rank: int


class Board(BaseModel):
    b0: Card
    b1: Card
    b2: Card
    b3: Card
    b4: Card


class Table(BaseModel):
    """"
    ante:   0.0
    small_blind:   0.0032992411870509386
    big_blind:   0.006598482374101877
    min_raise:   0.013196964748203754
    pot_amt:   0.0
    total_to_call:   0.006598482374101877
    p0_acts_next:   1.0
    p1_acts_next:   0.0
    p2_acts_next:   0.0
    p3_acts_next:   0.0
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
    """
    ante: float
    small_blind: float
    big_blind: float
    min_raise: float
    pot_amt: float
    total_to_call: float
    p0_acts_next: float
    p1_acts_next: float
    p2_acts_next: float
    p3_acts_next: float
    p4_acts_next: float
    p5_acts_next: float
    round_preflop: float
    round_flop: float
    round_turn: float
    round_river: float
    side_pot_0: float
    side_pot_1: float
    side_pot_2: float
    side_pot_3: float
    side_pot_4: float
    side_pot_5: float


class PlayerInfo(BaseModel):
    """"
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
    """
    pid: int
    stack_p: float
    curr_bet_p: float
    has_folded_this_episode_p: bool
    is_allin_p: bool
    side_pot_rank_p_is_0: int
    side_pot_rank_p_is_1: int
    side_pot_rank_p_is_2: int
    side_pot_rank_p_is_3: int
    side_pot_rank_p_is_4: int
    side_pot_rank_p_is_5: int
    c0: Card
    c1: Card


class LastAction(BaseModel):
    action_what: int
    action_how_much: float


class Info(BaseModel):
    continue_round: bool
    next_to_act: int  # only if continue_round = true
    rundown: bool
    next_round: bool
    # payouts: todo


class EnvState(BaseModel):
    # meta
    env_id: int
    n_players: int
    starting_stack_size: int
    # game
    table: Table
    p0: PlayerInfo
    p1: PlayerInfo
    p2: PlayerInfo
    p3: PlayerInfo
    p4: PlayerInfo
    p5: PlayerInfo
    board: Board
    # utils
    last_action: Optional[LastAction]
    human_player_index: Optional[int]
    human_player: Optional[str]
    done: bool
    # info: Info


class EnvironmentState(BaseModel):
    """
    ante:   0.0
    small_blind:   0.0032992411870509386
    big_blind:   0.006598482374101877
    min_raise:   0.013196964748203754
    pot_amt:   0.0
    total_to_call:   0.006598482374101877
    p0_acts_next:   1.0
    p1_acts_next:   0.0
    p2_acts_next:   0.0
    p3_acts_next:   0.0
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
    """
    ante: float
    small_blind: float
    big_blind: float
    min_raise: float
    pot_amt: float
    total_to_call: float
    p0_acts_next: float
    p1_acts_next: float
    p2_acts_next: float
    p3_acts_next: float
    p4_acts_next: float
    p5_acts_next: float
    round_preflop: float
    round_flop: float
    round_turn: float
    round_river: float
    side_pot_0: float
    side_pot_1: float
    side_pot_2: float
    side_pot_3: float
    side_pot_4: float
    side_pot_5: float
    stack_p0: float
    curr_bet_p0: float
    has_folded_this_episode_p0: float
    is_allin_p0: float
    side_pot_rank_p0_is_0: float
    side_pot_rank_p0_is_1: float
    side_pot_rank_p0_is_2: float
    side_pot_rank_p0_is_3: float
    side_pot_rank_p0_is_4: float
    side_pot_rank_p0_is_5: float
    stack_p1: float
    curr_bet_p1: float
    has_folded_this_episode_p1: float
    is_allin_p1: float
    side_pot_rank_p1_is_0: float
    side_pot_rank_p1_is_1: float
    side_pot_rank_p1_is_2: float
    side_pot_rank_p1_is_3: float
    side_pot_rank_p1_is_4: float
    side_pot_rank_p1_is_5: float
    stack_p2: float
    curr_bet_p2: float
    has_folded_this_episode_p2: float
    is_allin_p2: float
    side_pot_rank_p2_is_0: float
    side_pot_rank_p2_is_1: float
    side_pot_rank_p2_is_2: float
    side_pot_rank_p2_is_3: float
    side_pot_rank_p2_is_4: float
    side_pot_rank_p2_is_5: float
    stack_p3: float
    curr_bet_p3: float
    has_folded_this_episode_p3: float
    is_allin_p3: float
    side_pot_rank_p3_is_0: float
    side_pot_rank_p3_is_1: float
    side_pot_rank_p3_is_2: float
    side_pot_rank_p3_is_3: float
    side_pot_rank_p3_is_4: float
    side_pot_rank_p3_is_5: float
    stack_p4: float
    curr_bet_p4: float
    has_folded_this_episode_p4: float
    is_allin_p4: float
    side_pot_rank_p4_is_0: float
    side_pot_rank_p4_is_1: float
    side_pot_rank_p4_is_2: float
    side_pot_rank_p4_is_3: float
    side_pot_rank_p4_is_4: float
    side_pot_rank_p4_is_5: float
    stack_p5: float
    curr_bet_p5: float
    has_folded_this_episode_p5: float
    is_allin_p5: float
    side_pot_rank_p5_is_0: float
    side_pot_rank_p5_is_1: float
    side_pot_rank_p5_is_2: float
    side_pot_rank_p5_is_3: float
    side_pot_rank_p5_is_4: float
    side_pot_rank_p5_is_5: float
    first_board_card_rank_0: float
    first_board_card_rank_1: float
    first_board_card_rank_2: float
    first_board_card_rank_3: float
    first_board_card_rank_4: float
    first_board_card_rank_5: float
    first_board_card_rank_6: float
    first_board_card_rank_7: float
    first_board_card_rank_8: float
    first_board_card_rank_9: float
    first_board_card_rank_10: float
    first_board_card_rank_11: float
    first_board_card_rank_12: float
    first_board_card_suit_0: float
    first_board_card_suit_1: float
    first_board_card_suit_2: float
    first_board_card_suit_3: float
    second_board_card_rank_0: float
    second_board_card_rank_1: float
    second_board_card_rank_2: float
    second_board_card_rank_3: float
    second_board_card_rank_4: float
    second_board_card_rank_5: float
    second_board_card_rank_6: float
    second_board_card_rank_7: float
    second_board_card_rank_8: float
    second_board_card_rank_9: float
    second_board_card_rank_10: float
    second_board_card_rank_11: float
    second_board_card_rank_12: float
    second_board_card_suit_0: float
    second_board_card_suit_1: float
    second_board_card_suit_2: float
    second_board_card_suit_3: float
    third_board_card_rank_0: float
    third_board_card_rank_1: float
    third_board_card_rank_2: float
    third_board_card_rank_3: float
    third_board_card_rank_4: float
    third_board_card_rank_5: float
    third_board_card_rank_6: float
    third_board_card_rank_7: float
    third_board_card_rank_8: float
    third_board_card_rank_9: float
    third_board_card_rank_10: float
    third_board_card_rank_11: float
    third_board_card_rank_12: float
    third_board_card_suit_0: float
    third_board_card_suit_1: float
    third_board_card_suit_2: float
    third_board_card_suit_3: float
    fourth_board_card_rank_0: float
    fourth_board_card_rank_1: float
    fourth_board_card_rank_2: float
    fourth_board_card_rank_3: float
    fourth_board_card_rank_4: float
    fourth_board_card_rank_5: float
    fourth_board_card_rank_6: float
    fourth_board_card_rank_7: float
    fourth_board_card_rank_8: float
    fourth_board_card_rank_9: float
    fourth_board_card_rank_10: float
    fourth_board_card_rank_11: float
    fourth_board_card_rank_12: float
    fourth_board_card_suit_0: float
    fourth_board_card_suit_1: float
    fourth_board_card_suit_2: float
    fourth_board_card_suit_3: float
    fifth_board_card_rank_0: float
    fifth_board_card_rank_1: float
    fifth_board_card_rank_2: float
    fifth_board_card_rank_3: float
    fifth_board_card_rank_4: float
    fifth_board_card_rank_5: float
    fifth_board_card_rank_6: float
    fifth_board_card_rank_7: float
    fifth_board_card_rank_8: float
    fifth_board_card_rank_9: float
    fifth_board_card_rank_10: float
    fifth_board_card_rank_11: float
    fifth_board_card_rank_12: float
    fifth_board_card_suit_0: float
    fifth_board_card_suit_1: float
    fifth_board_card_suit_2: float
    fifth_board_card_suit_3: float
    first_player_card_0_rank_0: float
    first_player_card_0_rank_1: float
    first_player_card_0_rank_2: float
    first_player_card_0_rank_3: float
    first_player_card_0_rank_4: float
    first_player_card_0_rank_5: float
    first_player_card_0_rank_6: float
    first_player_card_0_rank_7: float
    first_player_card_0_rank_8: float
    first_player_card_0_rank_9: float
    first_player_card_0_rank_10: float
    first_player_card_0_rank_11: float
    first_player_card_0_rank_12: float
    first_player_card_0_suit_0: float
    first_player_card_0_suit_1: float
    first_player_card_0_suit_2: float
    first_player_card_0_suit_3: float
    first_player_card_1_rank_0: float
    first_player_card_1_rank_1: float
    first_player_card_1_rank_2: float
    first_player_card_1_rank_3: float
    first_player_card_1_rank_4: float
    first_player_card_1_rank_5: float
    first_player_card_1_rank_6: float
    first_player_card_1_rank_7: float
    first_player_card_1_rank_8: float
    first_player_card_1_rank_9: float
    first_player_card_1_rank_10: float
    first_player_card_1_rank_11: float
    first_player_card_1_rank_12: float
    first_player_card_1_suit_0: float
    first_player_card_1_suit_1: float
    first_player_card_1_suit_2: float
    first_player_card_1_suit_3: float
    second_player_card_0_rank_0: float
    second_player_card_0_rank_1: float
    second_player_card_0_rank_2: float
    second_player_card_0_rank_3: float
    second_player_card_0_rank_4: float
    second_player_card_0_rank_5: float
    second_player_card_0_rank_6: float
    second_player_card_0_rank_7: float
    second_player_card_0_rank_8: float
    second_player_card_0_rank_9: float
    second_player_card_0_rank_10: float
    second_player_card_0_rank_11: float
    second_player_card_0_rank_12: float
    second_player_card_0_suit_0: float
    second_player_card_0_suit_1: float
    second_player_card_0_suit_2: float
    second_player_card_0_suit_3: float
    second_player_card_1_rank_0: float
    second_player_card_1_rank_1: float
    second_player_card_1_rank_2: float
    second_player_card_1_rank_3: float
    second_player_card_1_rank_4: float
    second_player_card_1_rank_5: float
    second_player_card_1_rank_6: float
    second_player_card_1_rank_7: float
    second_player_card_1_rank_8: float
    second_player_card_1_rank_9: float
    second_player_card_1_rank_10: float
    second_player_card_1_rank_11: float
    second_player_card_1_rank_12: float
    second_player_card_1_suit_0: float
    second_player_card_1_suit_1: float
    second_player_card_1_suit_2: float
    second_player_card_1_suit_3: float
    third_player_card_0_rank_0: float
    third_player_card_0_rank_1: float
    third_player_card_0_rank_2: float
    third_player_card_0_rank_3: float
    third_player_card_0_rank_4: float
    third_player_card_0_rank_5: float
    third_player_card_0_rank_6: float
    third_player_card_0_rank_7: float
    third_player_card_0_rank_8: float
    third_player_card_0_rank_9: float
    third_player_card_0_rank_10: float
    third_player_card_0_rank_11: float
    third_player_card_0_rank_12: float
    third_player_card_0_suit_0: float
    third_player_card_0_suit_1: float
    third_player_card_0_suit_2: float
    third_player_card_0_suit_3: float
    third_player_card_1_rank_0: float
    third_player_card_1_rank_1: float
    third_player_card_1_rank_2: float
    third_player_card_1_rank_3: float
    third_player_card_1_rank_4: float
    third_player_card_1_rank_5: float
    third_player_card_1_rank_6: float
    third_player_card_1_rank_7: float
    third_player_card_1_rank_8: float
    third_player_card_1_rank_9: float
    third_player_card_1_rank_10: float
    third_player_card_1_rank_11: float
    third_player_card_1_rank_12: float
    third_player_card_1_suit_0: float
    third_player_card_1_suit_1: float
    third_player_card_1_suit_2: float
    third_player_card_1_suit_3: float
    fourth_player_card_0_rank_0: float
    fourth_player_card_0_rank_1: float
    fourth_player_card_0_rank_2: float
    fourth_player_card_0_rank_3: float
    fourth_player_card_0_rank_4: float
    fourth_player_card_0_rank_5: float
    fourth_player_card_0_rank_6: float
    fourth_player_card_0_rank_7: float
    fourth_player_card_0_rank_8: float
    fourth_player_card_0_rank_9: float
    fourth_player_card_0_rank_10: float
    fourth_player_card_0_rank_11: float
    fourth_player_card_0_rank_12: float
    fourth_player_card_0_suit_0: float
    fourth_player_card_0_suit_1: float
    fourth_player_card_0_suit_2: float
    fourth_player_card_0_suit_3: float
    fourth_player_card_1_rank_0: float
    fourth_player_card_1_rank_1: float
    fourth_player_card_1_rank_2: float
    fourth_player_card_1_rank_3: float
    fourth_player_card_1_rank_4: float
    fourth_player_card_1_rank_5: float
    fourth_player_card_1_rank_6: float
    fourth_player_card_1_rank_7: float
    fourth_player_card_1_rank_8: float
    fourth_player_card_1_rank_9: float
    fourth_player_card_1_rank_10: float
    fourth_player_card_1_rank_11: float
    fourth_player_card_1_rank_12: float
    fourth_player_card_1_suit_0: float
    fourth_player_card_1_suit_1: float
    fourth_player_card_1_suit_2: float
    fourth_player_card_1_suit_3: float
    fifth_player_card_0_rank_0: float
    fifth_player_card_0_rank_1: float
    fifth_player_card_0_rank_2: float
    fifth_player_card_0_rank_3: float
    fifth_player_card_0_rank_4: float
    fifth_player_card_0_rank_5: float
    fifth_player_card_0_rank_6: float
    fifth_player_card_0_rank_7: float
    fifth_player_card_0_rank_8: float
    fifth_player_card_0_rank_9: float
    fifth_player_card_0_rank_10: float
    fifth_player_card_0_rank_11: float
    fifth_player_card_0_rank_12: float
    fifth_player_card_0_suit_0: float
    fifth_player_card_0_suit_1: float
    fifth_player_card_0_suit_2: float
    fifth_player_card_0_suit_3: float
    fifth_player_card_1_rank_0: float
    fifth_player_card_1_rank_1: float
    fifth_player_card_1_rank_2: float
    fifth_player_card_1_rank_3: float
    fifth_player_card_1_rank_4: float
    fifth_player_card_1_rank_5: float
    fifth_player_card_1_rank_6: float
    fifth_player_card_1_rank_7: float
    fifth_player_card_1_rank_8: float
    fifth_player_card_1_rank_9: float
    fifth_player_card_1_rank_10: float
    fifth_player_card_1_rank_11: float
    fifth_player_card_1_rank_12: float
    fifth_player_card_1_suit_0: float
    fifth_player_card_1_suit_1: float
    fifth_player_card_1_suit_2: float
    fifth_player_card_1_suit_3: float
    sixth_player_card_0_rank_0: float
    sixth_player_card_0_rank_1: float
    sixth_player_card_0_rank_2: float
    sixth_player_card_0_rank_3: float
    sixth_player_card_0_rank_4: float
    sixth_player_card_0_rank_5: float
    sixth_player_card_0_rank_6: float
    sixth_player_card_0_rank_7: float
    sixth_player_card_0_rank_8: float
    sixth_player_card_0_rank_9: float
    sixth_player_card_0_rank_10: float
    sixth_player_card_0_rank_11: float
    sixth_player_card_0_rank_12: float
    sixth_player_card_0_suit_0: float
    sixth_player_card_0_suit_1: float
    sixth_player_card_0_suit_2: float
    sixth_player_card_0_suit_3: float
    sixth_player_card_1_rank_0: float
    sixth_player_card_1_rank_1: float
    sixth_player_card_1_rank_2: float
    sixth_player_card_1_rank_3: float
    sixth_player_card_1_rank_4: float
    sixth_player_card_1_rank_5: float
    sixth_player_card_1_rank_6: float
    sixth_player_card_1_rank_7: float
    sixth_player_card_1_rank_8: float
    sixth_player_card_1_rank_9: float
    sixth_player_card_1_rank_10: float
    sixth_player_card_1_rank_11: float
    sixth_player_card_1_rank_12: float
    sixth_player_card_1_suit_0: float
    sixth_player_card_1_suit_1: float
    sixth_player_card_1_suit_2: float
    sixth_player_card_1_suit_3: float
    human_player_position: int
    # human_player_card_00: str
    # human_player_card_01: str
    done: bool
