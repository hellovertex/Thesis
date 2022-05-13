from __future__ import annotations
from random import randint
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel
from starlette.requests import Request

from src.calls.environment.utils import get_table_info, get_board_cards, get_player_stats
from src.model.environment_state import EnvState, Info

router = APIRouter()
abbrevs = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth']


class Stacks(BaseModel):
    stack_p0: Optional[int]
    stack_p1: Optional[int]
    stack_p2: Optional[int]
    stack_p3: Optional[int]
    stack_p4: Optional[int]
    stack_p5: Optional[int]


class EnvironmentResetRequestBody(BaseModel):
    env_id: int
    stack_sizes: Optional[Stacks]

    class Config:
        schema_extra = {
            "env_id": {
                "example": 1,
                "description": "The environment unique id "
                               "used for requesting this specific environment."
            },
            "stack_sizes": {
                "example": 20000,
                "description": "The number of chips each player will get on resetting the environment."
                               "Note that the environment is reset on each hand dealt. This implies"
                               "that starting stacks can vary between players, e.g. in the middle of the game."
            }
        }


@router.post("/environment/{env_id}/reset/",
             response_model=EnvState,
             operation_id="reset_environment")
async def reset_environment(body: EnvironmentResetRequestBody, request: Request):
    env_id = body.env_id
    n_players = request.app.backend.active_ens[env_id].env.N_SEATS

    if request.app.backend.metadata[env_id]['initial_state']:
        human_player_position = randint(0, n_players - 1)
        request.app.backend.metadata[env_id]['initial_state'] = False
        request.app.backend.metadata[env_id]['human_player_position'] = human_player_position
    else:
        human_player_position = min(0, request.app.backend.metadata[env_id]['human_player_position'] - 1)
        request.app.backend.metadata[env_id]['human_player_position'] = human_player_position
    stack_sizes = body.stack_sizes
    if body.stack_sizes is None:
        default_stack = request.app.backend.active_ens[env_id].env.DEFAULT_STACK_SIZE
        stack_sizes = dict([(f'stack_p{i}', default_stack) for i in range(n_players)])
    obs, _, _, _ = request.app.backend.active_ens[env_id].reset()

    obs_dict = request.app.backend.active_ens[env_id].obs_idx_dict
    obs_keys = [k for k in obs_dict.keys()]

    table_info = get_table_info(obs_keys, obs)
    idx_end_table = obs_keys.index('side_pot_5')

    board_cards = get_board_cards(idx_board_start=obs_keys.index('0th_board_card_rank_0'),
                                  idx_board_end=obs_keys.index('0th_player_card_0_rank_0'),
                                  obs=obs)
    player_info = get_player_stats(obs_keys, obs, start_idx=idx_end_table + 1)

    result = {'env_id': env_id,
              'n_players': n_players,
              'stack_sizes': stack_sizes,
              'last_action': None,
              'table': table_info,
              **player_info,
              'board': board_cards,
              'human_player_index': human_player_position,
              'human_player': ['p0', 'p1', 'p2', 'p3', 'p4', 'p5'][human_player_position],
              'p_acts_next': request.app.backend.active_ens[env_id].env.current_player.seat_id,
              'done': False,
              'info': Info(**{'continue_round': True,
                              'draw_next_stage': False,
                              'rundown': False,
                              'deal_next_hand': False,
                              'payouts': None})
              }
    return EnvState(**dict(result))
