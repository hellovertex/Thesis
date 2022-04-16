from fastapi import APIRouter
from starlette.requests import Request

from src.model.environment_state import EnvironmentState

router = APIRouter()


def maybe_replace_leading_digit(val):
    val = val.replace('0th', 'first')
    val = val.replace('1th', 'second')
    val = val.replace('2th', 'third')
    val = val.replace('3th', 'fourth')
    val = val.replace('4th', 'fifth')
    return val.replace('5th', 'sixth')


@router.get("/environment/{env_id}/reset",
            response_model=EnvironmentState,
            operation_id="reset_environment")
async def reset_environment(request: Request, env_id: int):
    obs, _, _, _ = request.app.backend.active_ens[env_id].reset()
    obs_keys = request.app.backend.active_ens[env_id].obs_idx_dict.keys()
    obs_keys = [maybe_replace_leading_digit(k) for k in obs_keys]

    last_relevant_index = obs_keys.index('sixth_player_card_1_suit_3')
    result = list(zip(obs_keys, obs))[:last_relevant_index + 1]

    return EnvironmentState(**dict(result))
