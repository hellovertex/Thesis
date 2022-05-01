import re

import numpy as np

from src.model.environment_state import PlayerInfo, Card, Board, TableInfo


def maybe_replace_leading_digit(val):
    val = val.replace('0th', 'first')
    val = val.replace('1th', 'second')
    val = val.replace('2th', 'third')
    val = val.replace('3th', 'fourth')
    val = val.replace('4th', 'fifth')
    return val.replace('5th', 'sixth')


def get_player_cards(idx_start, idx_end, obs, n_suits=4, n_ranks=13):
    cur_idx = idx_start
    cards = {}
    end_idx = 0
    for i in range(2):
        suit = -1
        rank = -1
        end_idx = cur_idx + n_suits + n_ranks
        bits = obs[cur_idx:end_idx]
        # print(f'obs[cur_idx:end_idx] = {obs[cur_idx:end_idx]}')
        if sum(bits) > 0:
            idx = np.where(bits == 1)[0]
            rank, suit = idx[0], idx[1]

        cards[f'c{i}'] = Card(**{'name': f'c{i}',
                                 'suit': suit,
                                 'rank': rank})
        cur_idx = end_idx
    assert end_idx == idx_end
    return cards


def get_player_stats(obs_keys, obs, start_idx):
    # cards
    cp0 = get_player_cards(idx_start=obs_keys.index("0th_player_card_0_rank_0"),
                           idx_end=obs_keys.index("1th_player_card_0_rank_0"),
                           obs=obs)
    cp1 = get_player_cards(idx_start=obs_keys.index("1th_player_card_0_rank_0"),
                           idx_end=obs_keys.index("2th_player_card_0_rank_0"),
                           obs=obs)
    cp2 = get_player_cards(idx_start=obs_keys.index("2th_player_card_0_rank_0"),
                           idx_end=obs_keys.index("3th_player_card_0_rank_0"),
                           obs=obs)
    cp3 = get_player_cards(idx_start=obs_keys.index("3th_player_card_0_rank_0"),
                           idx_end=obs_keys.index("4th_player_card_0_rank_0"),
                           obs=obs)
    cp4 = get_player_cards(idx_start=obs_keys.index("4th_player_card_0_rank_0"),
                           idx_end=obs_keys.index("5th_player_card_0_rank_0"),
                           obs=obs)
    cp5 = get_player_cards(idx_start=obs_keys.index("5th_player_card_0_rank_0"),
                           idx_end=obs_keys.index("preflop_player_0_action_0_how_much"),
                           obs=obs)
    # stats
    idx_end_p0 = obs_keys.index('side_pot_rank_p0_is_5') + 1
    idx_end_p1 = obs_keys.index('side_pot_rank_p1_is_5') + 1
    idx_end_p2 = obs_keys.index('side_pot_rank_p2_is_5') + 1
    idx_end_p3 = obs_keys.index('side_pot_rank_p3_is_5') + 1
    idx_end_p4 = obs_keys.index('side_pot_rank_p4_is_5') + 1
    idx_end_p5 = obs_keys.index('side_pot_rank_p5_is_5') + 1
    obs_keys = [re.sub(re.compile(r'p\d'), 'p', s) for s in obs_keys]

    p0 = list(zip(obs_keys, obs))[start_idx:idx_end_p0]
    p1 = list(zip(obs_keys, obs))[idx_end_p0:idx_end_p1]
    p2 = list(zip(obs_keys, obs))[idx_end_p1:idx_end_p2]
    p3 = list(zip(obs_keys, obs))[idx_end_p2:idx_end_p3]
    p4 = list(zip(obs_keys, obs))[idx_end_p3:idx_end_p4]
    p5 = list(zip(obs_keys, obs))[idx_end_p4:idx_end_p5]

    return {'p0': PlayerInfo(**{'pid': 0, **dict(p0), **dict(cp0)}),
            'p1': PlayerInfo(**{'pid': 1, **dict(p1), **dict(cp1)}),
            'p2': PlayerInfo(**{'pid': 2, **dict(p2), **dict(cp2)}),
            'p3': PlayerInfo(**{'pid': 3, **dict(p3), **dict(cp3)}),
            'p4': PlayerInfo(**{'pid': 4, **dict(p4), **dict(cp4)}),
            'p5': PlayerInfo(**{'pid': 5, **dict(p5), **dict(cp5)})}


def get_board_cards(idx_board_start, idx_board_end, obs, n_suits=4, n_ranks=13):
    cur_idx = idx_board_start
    cards = {}
    end_idx = 0
    for i in range(5):
        suit = -1
        rank = -1
        end_idx = cur_idx + n_suits + n_ranks
        bits = obs[cur_idx:end_idx]
        if sum(bits) > 0:
            idx = np.where(bits == 1)[0]
            rank, suit = idx[0], idx[1]

        cards[f'b{i}'] = Card(**{'name': f'b{i}',
                                 'suit': suit,
                                 'rank': rank})
        cur_idx = end_idx
    # print(f'idx_board_end = {idx_board_end}')
    # print(f'end_idx = {end_idx}')
    assert idx_board_end == end_idx
    return Board(**cards)


def get_table_info(obs_keys, obs):
    table_kwargs = list(zip(obs_keys, obs))[0:obs_keys.index('side_pot_5') + 1]
    return TableInfo(**dict(table_kwargs))
