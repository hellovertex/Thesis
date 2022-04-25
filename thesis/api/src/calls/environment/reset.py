from random import randint

from fastapi import APIRouter
from starlette.requests import Request

from src.model.environment_state import EnvironmentState
from .utils import maybe_replace_leading_digit

router = APIRouter()


@router.get("/environment/{env_id}/reset",
            response_model=EnvironmentState,
            operation_id="reset_environment")
async def reset_environment(request: Request, env_id: int):
    human_player_position = randint(0, request.app.backend.active_ens[env_id].env.N_SEATS)
    obs, _, _, _ = request.app.backend.active_ens[env_id].reset()
    obs_keys = request.app.backend.active_ens[env_id].obs_idx_dict.keys()
    obs_keys = [maybe_replace_leading_digit(k) for k in obs_keys]

    last_relevant_index = obs_keys.index('sixth_player_card_1_suit_3')
    result = list(zip(obs_keys, obs))[:last_relevant_index + 1]
    result.append(('human_player_position', human_player_position))
    result.append(('done', False))
    return EnvironmentState(**dict(result))
