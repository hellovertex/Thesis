from typing import Optional, List, Dict, Any

from PokerRL import NoLimitHoldem
from PokerRL.game.wrappers import AugmentObservationWrapper


class Backend:
    def __init__(self):
        self._num_active_environments = 0
        self.active_ens: Optional[Dict[int, Any]] = {}
        self.metadata: Optional[Dict[int, Dict]] = {}

    def make_environment(self, config: dict):
        self._num_active_environments += 1
        env_id = self._num_active_environments
        num_players = config['n_players']
        starting_stack_sizes = [config['starting_stack_size'] for _ in range(num_players)]
        args = NoLimitHoldem.ARGS_CLS(n_seats=num_players,
                                      starting_stack_sizes_list=starting_stack_sizes,
                                      use_simplified_headsup_obs=False)
        env = NoLimitHoldem(is_evaluating=True,
                            env_args=args,
                            lut_holder=NoLimitHoldem.get_lut_holder())
        # todo install env + wrapper as a separate package
        env_wrapped = AugmentObservationWrapper(env)

        self.active_ens[env_id] = env_wrapped
        self.metadata[env_id] = {'initial_state': True}
        return env_id
