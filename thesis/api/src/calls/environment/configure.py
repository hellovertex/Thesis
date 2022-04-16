from fastapi import APIRouter

from PokerRL import NoLimitHoldem
from src.model.environment_config import EnvironmentConfig
from backend import backend

router = APIRouter()


@router.post("/environment/configure",
             response_model=EnvironmentConfig,
             operation_id="configure_environment")
async def configure_environment(n_players: int, starting_stack_size: int):
    # make args for env
    config = {"n_players": n_players,
              "starting_stack_size": starting_stack_size}

    return EnvironmentConfig(env_id=backend.make_environment(config),
                             num_players=n_players,
                             starting_stack_size=starting_stack_size)
