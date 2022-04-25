from fastapi import APIRouter
from starlette.requests import Request

from src.model.environment_state import EnvironmentState
from .utils import maybe_replace_leading_digit

router = APIRouter()


@router.get("/environment/{env_id}/step",
            response_model=EnvironmentState,
            operation_id="step_environment")
async def step_environment(request: Request,
                           env_id: int,
                           action: int,
                           action_how_much: float):
    action = (action, action_how_much)
    obs, _, done, _ = request.app.backend.active_ens[env_id].step(action)
    obs_keys = request.app.backend.active_ens[env_id].obs_idx_dict.keys()
    obs_keys = [maybe_replace_leading_digit(k) for k in obs_keys]

    last_relevant_index = obs_keys.index('sixth_player_card_1_suit_3')
    result = list(zip(obs_keys, obs))[:last_relevant_index + 1]
    result.append(('human_player_position', -1))  # invalid because is set in reset
    result.append(('done', done))
    return EnvironmentState(**dict(result))
