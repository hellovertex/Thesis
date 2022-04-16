import numpy as np
from PokerRL import NoLimitHoldem
from src.env_package_mock.env_wrapper import AugmentObservationWrapper


def test_encode_next_player():
    args = NoLimitHoldem.ARGS_CLS(n_seats=6,
                                  starting_stack_sizes_list=[20000 for _ in range(6)],
                                  use_simplified_headsup_obs=False)
    env = NoLimitHoldem(is_evaluating=True,
                        env_args=args,
                        lut_holder=NoLimitHoldem.get_lut_holder())
    env = AugmentObservationWrapper(env)
    obs, _, _, _ = env.reset()
    start = env.obs_idx_dict['p0_acts_next']
    end = env.obs_idx_dict['p5_acts_next'] + 1
    assert sum(obs[start:end]) == 1

    args = NoLimitHoldem.ARGS_CLS(n_seats=2,
                                  starting_stack_sizes_list=[20000 for _ in range(6)],
                                  use_simplified_headsup_obs=False)
    env = NoLimitHoldem(is_evaluating=True,
                        env_args=args,
                        lut_holder=NoLimitHoldem.get_lut_holder())
    env = AugmentObservationWrapper(env)
    obs, _, _, _ = env.reset()
    start = env.obs_idx_dict['p0_acts_next']
    end = env.obs_idx_dict['p5_acts_next'] + 1
    assert sum(obs[start:end]) == 1


def test_zero_padding_in_between():
    max_players = 6
    num_players = 3
    arr = np.array([1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0])
    bits_per_player = np.split(arr, 3)  # [array([1, 1, 1, 0]), array([1, 1, 1, 0]), array([1, 1, 1, 0])]
    bits_to_pad_in_between = np.full(max_players - num_players, 4)  # [4 4 4]
    padded_in_between = np.array([np.append(s, bits_to_pad_in_between) for s in bits_per_player])
    padded_in_netween = np.hstack(padded_in_between)
    assert np.array_equal(padded_in_netween,
                          np.array([1, 1, 1, 0, 4, 4, 4, 1, 1, 1, 0, 4, 4, 4, 1, 1, 1, 0, 4, 4, 4]))
