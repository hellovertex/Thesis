# Copyright (c) 2019 Eric Steinberger


import copy
import unittest
from unittest import TestCase

import numpy as np

from PokerRL.game.games import NoLimitHoldem


class TestPokerEnv(TestCase):
    ITERATIONS = 30
    MIN_P = 2
    MAX_P = 6
    DO_RENDER = False

    def test_get_current_obs(self):
        args = NoLimitHoldem.ARGS_CLS(n_seats=3,
                                      stack_randomization_range=(0, 0),
                                      starting_stack_sizes_list=[1000] * 3)
        env = NoLimitHoldem(is_evaluating=False, env_args=args, lut_holder=NoLimitHoldem.get_lut_holder())
        env.reset()
        env.step([1, -1])
        a = env.get_current_obs(is_terminal=False)
        b = env.get_current_obs(is_terminal=False)
        # print(env.print_obs(a))
        # print(env.deck.state_dict())
        assert np.array_equal(a, b)

        # terminal should be all 0
        assert np.array_equal(np.zeros_like(a), env.get_current_obs(is_terminal=True))


def _get_new_nlh_env(n_seats, min_stack=100, max_stack=1000, random_stacks=False):
    r_m = 0
    if random_stacks:
        r_m = min_stack - max_stack
    args = NoLimitHoldem.ARGS_CLS(n_seats=n_seats,
                                  stack_randomization_range=(r_m, 0),
                                  starting_stack_sizes_list=[1000] * n_seats)
    return NoLimitHoldem(env_args=args, is_evaluating=True, lut_holder=NoLimitHoldem.get_lut_holder())


if __name__ == '__main__':
    unittest.main()
