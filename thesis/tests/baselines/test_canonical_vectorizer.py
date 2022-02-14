# [1 1 1 0 4 4 4 1 1 1 0 4 4 4 1 1 1 0 4 4 4]
import numpy as np


def test_zero_padding_in_between():
    max_players = 6
    num_players = 3
    arr = np.array([1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0])
    bits_per_player = np.split(arr, 3)  # [array([1, 1, 1, 0]), array([1, 1, 1, 0]), array([1, 1, 1, 0])]
    bits_to_pad_in_between = np.full(max_players-num_players, 4)  # [4 4 4]
    padded_in_between = np.array([np.append(s, bits_to_pad_in_between) for s in bits_per_player])
    padded_in_netween = np.hstack(padded_in_between)
    assert np.array_equal(padded_in_netween,
                          np.array([1, 1, 1, 0, 4, 4, 4, 1, 1, 1, 0, 4, 4, 4, 1, 1, 1, 0, 4, 4, 4]))
