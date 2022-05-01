import re
from random import randint

import numpy as np
from fastapi import APIRouter
from starlette.requests import Request

from src.model.environment_state import PlayerInfo, Card, Board, TableInfo, EnvState

router = APIRouter()
abbrevs = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth']


def get_player_stats(obs_keys, obs, start_idx):
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
    return {'p0': PlayerInfo(**{'pid': 0, **dict(p0)}),
            'p1': PlayerInfo(**{'pid': 1, **dict(p1)}),
            'p2': PlayerInfo(**{'pid': 2, **dict(p2)}),
            'p3': PlayerInfo(**{'pid': 3, **dict(p3)}),
            'p4': PlayerInfo(**{'pid': 4, **dict(p4)}),
            'p5': PlayerInfo(**{'pid': 5, **dict(p5)})}


def get_player_cards(player_index, obs_dict):
    suit = obs_dict
    # todo finish


def get_board_cards(idx_board_start, idx_board_end, obs, n_suits=4, n_ranks=13):
    cur_idx = idx_board_start
    cards = {}
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
    return Board(**cards)


def get_table_info(obs_keys, obs):
    table_kwargs = list(zip(obs_keys, obs))[0:obs_keys.index('side_pot_5') + 1]
    return TableInfo(**dict(table_kwargs))


@router.get("/environment/{env_id}/reset",
            # response_model=EnvironmentState,
            response_model=EnvState,
            operation_id="reset_environment")
async def reset_environment(request: Request, env_id: int):
    human_player_position = randint(0, request.app.backend.active_ens[env_id].env.N_SEATS)
    obs, _, _, _ = request.app.backend.active_ens[env_id].reset()
    obs_dict = request.app.backend.active_ens[env_id].obs_idx_dict
    obs_keys = obs_dict.keys()
    obs_keys = [k for k in obs_keys]
    # obs_keys = [maybe_replace_leading_digit(k) for k in obs_keys]

    # todo card0, card1 = get_player_cards(human_player_position, obs_dict)
    table_info = get_table_info(obs_keys, obs)
    idx_end_table = obs_keys.index('side_pot_5')

    idx_end_p5 = obs_keys.index('side_pot_rank_p5_is_5')
    board_cards = get_board_cards(idx_board_start=idx_end_p5 + 1,
                                  idx_board_end=obs_keys.index('4th_board_card_suit_3'),
                                  obs=obs)
    player_info = get_player_stats(obs_keys, obs, start_idx=idx_end_table + 1)
    # last_relevant_index = obs_keys.index('sixth_player_card_1_suit_3')
    # result = list(zip(obs_keys, obs))[:last_relevant_index + 1]
    result = {'table_info': table_info,
              **player_info,
              'board': board_cards,
              'human_player_position': human_player_position,
              'done': False
              }
    # result.append(('human_player_position', human_player_position))
    # result.append(('done', False))
    # return EnvironmentState(**dict(result))
    return EnvState(**dict(result))
