from __future__ import annotations
from random import randint
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel
from starlette.requests import Request
import numpy as np

from PokerRL import NoLimitHoldem
from src.calls.environment.utils import get_table_info, get_board_cards, get_player_stats
from src.model.environment_state import EnvironmentState, Info, Players

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
             response_model=EnvironmentState,
             operation_id="reset_environment")
async def reset_environment(body: EnvironmentResetRequestBody, request: Request):
    env_id = body.env_id
    number_non_null_stacks = [stack for stack in body.stack_sizes.dict().values() if stack is not None]
    # if no starting stacks are provided we can safely get number of players from environment configuration
    # otherwise, starting stacks provided by the client indicate a maybe reduced number of players
    n_players = request.app.backend.active_ens[env_id].env.N_SEATS if (body.stack_sizes is None) else len(
        number_non_null_stacks)

    # set button
    if request.app.backend.metadata[env_id]['initial_state']:
        # 1. randomly determine first button holder
        button_index = randint(0, n_players - 1)
        request.app.backend.metadata[env_id]['initial_state'] = False
        request.app.backend.metadata[env_id]['button_index'] = button_index
    else:
        # 2. move button +1 to the left
        request.app.backend.metadata[env_id]['button_index'] += 1
        if request.app.backend.metadata[env_id]['button_index'] == n_players:
            request.app.backend.metadata[env_id]['button_index'] = 0

    # set stack sizes
    stack_sizes = body.stack_sizes
    button_index = request.app.backend.metadata[env_id]['button_index']
    if body.stack_sizes is None:
        # 1. fall back to default stack size if no stacks were provided in request
        default_stack = request.app.backend.active_ens[env_id].env.DEFAULT_STACK_SIZE
        stack_sizes_dict = dict([(f'stack_p{i}', default_stack) for i in range(n_players)])
    else:
        # 2. set custom stack sizes provided in the request body
        if not request.app.backend.metadata[env_id]['initial_state']:
            # 2.1 roll stacks
            # only roll stacks, when not in initial state, which is sometimes the case when debugging where
            # we set custom stacks despite not having played before. In this case, rolling does not make sense
            stack_sizes = np.roll(list(body.stack_sizes.dict().values()), -button_index, axis=0)
            # stack_sizes = [stack.item() for stack in stack_sizes]
            stack_sizes_dict = dict(list(zip(body.stack_sizes.dict().keys(), stack_sizes)))
            # set env_args such that new starting stacks are used
        args = NoLimitHoldem.ARGS_CLS(n_seats=n_players,
                                      starting_stack_sizes_list=stack_sizes,
                                      use_simplified_headsup_obs=False)
        request.app.backend.active_ens[env_id].overwrite_args(args)
    obs, _, _, _ = request.app.backend.active_ens[env_id].reset()

    obs_dict = request.app.backend.active_ens[env_id].obs_idx_dict
    obs_keys = [k for k in obs_dict.keys()]

    table_info = get_table_info(obs_keys, obs, offset=button_index)
    idx_end_table = obs_keys.index('side_pot_5')

    board_cards = get_board_cards(idx_board_start=obs_keys.index('0th_board_card_rank_0'),
                                  idx_board_end=obs_keys.index('0th_player_card_0_rank_0'),
                                  obs=obs)
    player_info = get_player_stats(obs_keys, obs, start_idx=idx_end_table + 1, offset=button_index)

    # move everything relative to hero offset
    p_acts_next = request.app.backend.active_ens[env_id].env.current_player.seat_id
    pid = button_index + p_acts_next
    p_acts_next = pid if pid < n_players else pid - n_players

    result = {'env_id': env_id,
              'n_players': n_players,
              'stack_sizes': stack_sizes_dict,
              'last_action': None,
              'table': table_info,
              'players': player_info,
              'board': board_cards,
              'button_index': button_index,
              'p_acts_next': p_acts_next,
              'done': False,
              'info': Info(**{'continue_round': True,
                              'draw_next_stage': False,
                              'rundown': False,
                              'deal_next_hand': False,
                              'payouts': None})
              }
    return EnvironmentState(**dict(result))
