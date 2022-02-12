from txt_parser import TxtParser
from state_encoder import RLStateEncoder
from thesis.core.wrapper import AugmentObservationWrapper


def main(f_path: str):
    """Parses hand_database and returns vectorized observations as returned by rl_env."""
    parser = TxtParser()
    parsed_hands = parser.parse_file(f_path)
    # use AugmentedEnvBuilder to get augmented observations encodings
    enc = RLStateEncoder(env_wrapper_cls=AugmentObservationWrapper)

    for hand in parsed_hands:
        observations, actions = enc.encode_episode(hand)
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
