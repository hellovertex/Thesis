from fastapi import APIRouter
from pydantic import BaseModel
from starlette.requests import Request
import numpy as np
from src.model.environment_state import EnvironmentState, EnvironmentState, LastAction, Info, Players
from .utils import get_table_info, get_board_cards, get_player_stats

router = APIRouter()


class EnvironmentStepRequestBody(BaseModel):
    env_id: int
    action: int
    action_how_much: float


@router.post("/environment/{env_id}/step",
             response_model=EnvironmentState,
             operation_id="step_environment")
async def step_environment(body: EnvironmentStepRequestBody, request: Request):
    n_players = request.app.backend.active_ens[body.env_id].env.N_SEATS

    starting_stack_size = request.app.backend.active_ens[body.env_id].env.DEFAULT_STACK_SIZE
    if body.action == -1:  # query ai model
        action = (0, -1)
    else:
        action = (body.action, body.action_how_much)

    obs, a, done, info = request.app.backend.active_ens[body.env_id].step(action)
    offset = request.app.backend.metadata[body.env_id]['button_index']
    # if action was fold, but player could have checked, the environment internally changes the action
    # if that happens, we must overwrite last action accordingly
    action = request.app.backend.active_ens[body.env_id].env.last_action  # [what, how_much, who]
    action = action[0], action[1]  # drop who
    print(f'a = {a}')
    print(f'done = {done}')
    print(f'info = {info}')
    obs_dict = request.app.backend.active_ens[body.env_id].obs_idx_dict
    obs_keys = [k for k in obs_dict.keys()]

    table_info = get_table_info(obs_keys, obs, offset)
    idx_end_table = obs_keys.index('side_pot_5')

    board_cards = get_board_cards(idx_board_start=obs_keys.index('0th_board_card_rank_0'),
                                  idx_board_end=obs_keys.index('0th_player_card_0_rank_0'),
                                  obs=obs)
    player_info = get_player_stats(obs_keys, obs, start_idx=idx_end_table + 1, offset=offset, n_players=n_players)
    print(f'current_player = {request.app.backend.active_ens[body.env_id].env.current_player.seat_id}')
    seats = request.app.backend.active_ens[body.env_id].env.seats
    stack_sizes = dict([(f'stack_p{i}', seats[i].stack) for i in range(len(seats))])

    # move everything relative to hero offset
    stack_sizes_rolled = np.roll(list(stack_sizes.values()), offset, axis=0)
    stack_sizes_rolled = [s.item() for s in stack_sizes_rolled]
    stack_sizes_rolled = dict(list(zip(stack_sizes.keys(), stack_sizes_rolled)))
    payouts_rolled = {}
    for k,v in info['payouts'].items():
        pid = offset + k if offset + k < n_players else offset + k - n_players
        payouts_rolled[pid] = v

    # offset relative to hero
    p_acts_next = request.app.backend.active_ens[body.env_id].env.current_player.seat_id
    pid = offset + p_acts_next
    p_acts_next = pid if pid < n_players else pid - n_players

    # players_with_chips_left = [p if not p.is_all_in]
    result = {'env_id': body.env_id,
              'n_players': n_players,
              'stack_sizes': stack_sizes_rolled,
              'last_action': LastAction(**{'action_what': action[0], 'action_how_much': action[1]}),
              'table': table_info,
              'players': player_info,
              'board': board_cards,
              'button_index': offset,
              'done': done,
              # todo this jumps from 3 to 1 instead of going from 3 to 4
              'p_acts_next': p_acts_next,
              'info': Info(**{'continue_round': info['continue_round'],
                              'draw_next_stage': info['draw_next_stage'],
                              'rundown': info['rundown'],
                              'deal_next_hand': info['deal_next_hand'],
                              'payouts': payouts_rolled})
              }
    return EnvironmentState(**dict(result))
