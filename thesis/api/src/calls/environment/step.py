from fastapi import APIRouter
from pydantic import BaseModel
from starlette.requests import Request

from src.model.environment_state import EnvironmentState, EnvState, LastAction, Info
from .utils import get_table_info, get_board_cards, get_player_stats

router = APIRouter()


class EnvironmentStepRequestBody(BaseModel):
    env_id: int
    action: int
    action_how_much: float


@router.post("/environment/{env_id}/step",
             response_model=EnvState,
             operation_id="step_environment")
async def step_environment(body: EnvironmentStepRequestBody, request: Request):
    n_players = request.app.backend.active_ens[body.env_id].env.N_SEATS

    starting_stack_size = request.app.backend.active_ens[body.env_id].env.DEFAULT_STACK_SIZE
    if body.action == -1:  # query ai model
        action = (0, -1)
    else:
        action = (body.action, body.action_how_much)

    obs, a, done, info = request.app.backend.active_ens[body.env_id].step(action)
    print(f'a = {a}')
    print(f'done = {done}')
    print(f'info = {info}')
    obs_dict = request.app.backend.active_ens[body.env_id].obs_idx_dict
    obs_keys = [k for k in obs_dict.keys()]

    table_info = get_table_info(obs_keys, obs)
    idx_end_table = obs_keys.index('side_pot_5')

    board_cards = get_board_cards(idx_board_start=obs_keys.index('0th_board_card_rank_0'),
                                  idx_board_end=obs_keys.index('0th_player_card_0_rank_0'),
                                  obs=obs)
    player_info = get_player_stats(obs_keys, obs, start_idx=idx_end_table + 1)
    print(f'current_player = {request.app.backend.active_ens[body.env_id].env.current_player.seat_id}')
    seats = request.app.backend.active_ens[body.env_id].env.seats
    stack_sizes = [(f'stack_p{i}', seats[i].stack) for i in range(len(seats))]
    # players_with_chips_left = [p if not p.is_all_in]
    result = {'env_id': body.env_id,
              'n_players': n_players,
              'stack_sizes': stack_sizes,
              'last_action': LastAction(**{'action_what': action[0], 'action_how_much': action[1]}),
              'table': table_info,
              **player_info,
              'board': board_cards,
              'human_player_index': None,
              'human_player': None,
              'done': done,
              # todo this jumps from 3 to 1 instead of going from 3 to 4
              'p_acts_next': request.app.backend.active_ens[body.env_id].env.current_player.seat_id,
              # 'info': Info(**{'continue_round': True,
              #                 'draw_next_stage': False,
              #                 'rundown': False,
              #                 'deal_next_hand': False,
              #                 'payouts': None})
              'info': Info(**{'continue_round': info['continue_round'],
                              'draw_next_stage': info['draw_next_stage'],
                              'rundown': info['rundown'],
                              'deal_next_hand': info['deal_next_hand'],
                              'payouts': info['payouts']})
              }
    return EnvState(**dict(result))
