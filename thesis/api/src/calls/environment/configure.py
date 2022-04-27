from fastapi import APIRouter
from src.model.environment_config import EnvironmentConfig, EnvironmentConfigRequestBody
from starlette.requests import Request

@router.post("/environment/configure",
             response_model=EnvironmentConfig,
             operation_id="configure_environment")
async def configure_environment(request: Request, config: EnvironmentConfigRequestBody):
    """Creates an environment in the backend and returns its unique ID.
    Use this ID with /reset and /step endpoints to play the game.

    Internal: Calls backend.Backend.make_environment(...),
    returns its id and config wrapped in EnvironmentConfig Model class"""
    n_players = config.n_players
    starting_stack_size = config.starting_stack_size
    assert 2 <= n_players <= 6
    # make args for env
    print(n_players)
    print(starting_stack_size)
    config = {"n_players": n_players,
              "starting_stack_size": starting_stack_size}

    return EnvironmentConfig(env_id=request.app.backend.make_environment(config),
                             num_players=n_players,
                             starting_stack_size=starting_stack_size)


router = APIRouter()
