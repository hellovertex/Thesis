from typing import Optional, List, Dict, Any

from PokerRL import NoLimitHoldem
# from baselines.supervised_learning.data.steinberger_wrapper import AugmentObservationWrapper


class Backend:
    def __init__(self):
        self._num_active_environments = 0
        self._active_ens: List[Optional[Dict[int, Any]]] = []

    def make_environment(self, config: dict):
        self._num_active_environments += 1
        env_id = self._num_active_environments
        num_players = config['n_players']
        starting_stack_sizes = [config['starting_stack_size'] for _ in range(num_players)]
        args = NoLimitHoldem.ARGS_CLS(n_seats=num_players,
                                      starting_stack_sizes_list=starting_stack_sizes)
        env = NoLimitHoldem(is_evaluating=True,
                            env_args=args,
                            lut_holder=NoLimitHoldem.get_lut_holder())
        # env = AugmentObservationWrapper(env)

        self._active_ens.append({env_id: env})
        return env_id


backend = Backend()
