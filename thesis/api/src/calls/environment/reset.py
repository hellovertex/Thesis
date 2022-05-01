from random import randint

from fastapi import APIRouter
from starlette.requests import Request

from src.calls.environment.utils import get_table_info, get_board_cards, get_player_stats, get_player_cards
from src.model.environment_state import EnvState

router = APIRouter()
abbrevs = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth']


@router.get("/environment/{env_id}/reset",
            # response_model=EnvironmentState,
            response_model=EnvState,
            operation_id="reset_environment")
async def reset_environment(request: Request, env_id: int):
    human_player_position = randint(0, request.app.backend.active_ens[env_id].env.N_SEATS)
    obs, _, _, _ = request.app.backend.active_ens[env_id].reset()

    obs_dict = request.app.backend.active_ens[env_id].obs_idx_dict
    obs_keys = [k for k in obs_dict.keys()]

    table_info = get_table_info(obs_keys, obs)
    idx_end_table = obs_keys.index('side_pot_5')

    board_cards = get_board_cards(idx_board_start=obs_keys.index('0th_board_card_rank_0'),
                                  idx_board_end=obs_keys.index('0th_player_card_0_rank_0'),
                                  obs=obs)
    player_info = get_player_stats(obs_keys, obs, start_idx=idx_end_table + 1)

    result = {'table_info': table_info,
              **player_info,
              'board': board_cards,
              'human_player_index': human_player_position,
              'human_player': ['p0', 'p1', 'p2', 'p3', 'p4', 'p5'][human_player_position],
              'done': False
              }
    return EnvState(**dict(result))
