import logging

from neuron_poker.gym_env.env import HoldemTable, Action
from enum import Enum


class Stage(Enum):
    """Allowed actions"""

    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    END_HIDDEN = 4
    SHOWDOWN = 5


class PlayerForTest:
    """Player shell"""

    def __init__(self, stack_size=100, name='TestPlayer'):
        """Initiaization of an agent"""
        self.stack = stack_size
        self.seat = None
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.agent_obj = None

    @staticmethod
    def action(action, observation, info):
        """Perform action."""
        _ = (observation, info)
        return action


def _create_env(n_players):
    """Create an environment"""
    env = HoldemTable()
    for _ in range(n_players):
        player = PlayerForTest()
        env.add_player(player)
    env.reset()
    return env


def test_actions():
    # self.stage = wird in end_round() gesetzt
    # end_round wird in _next_player gecallt
    # wenn PlayerCycle.next_player() none returned
    # => PlayerCycle.next_player() ist key
    # line 607 is called when it shouldnt
    # must depend on max_raising_rounds
    # actually we wouldnt get there if legal moves was corrected
    """Test basic actions with 3 players, because I observed this sequence
    during debugging:
        WARNING:root:player: 0, action: Action.RAISE_POT
        WARNING:root:player: 1, action: Action.RAISE_POT
        WARNING:root:player: 2, action: Action.RAISE_POT
        WARNING:root:player: 0, action: Action.RAISE_POT
        WARNING:root:player: 1, action: Action.RAISE_POT
        WARNING:root:player: 1, action: Action.ALL_IN
        WARNING:root:player: 2, action: Action.FOLD
        WARNING:root:player: 0, action: Action.RAISE_3BB
    """
    env = _create_env(3)
    assert env.current_player.seat == 0
    assert env.player_cycle.idx == 0
    env.step(Action.RAISE_POT)  # seat 0 dealer
    assert env.current_player.seat == 1
    assert env.player_cycle.idx == 1
    env.step(Action.RAISE_POT)  # seat 1 sb
    assert env.current_player.seat == 2
    assert env.player_cycle.idx == 2
    env.step(Action.RAISE_POT)  # seat 2 bb
    assert env.current_player.seat == 0
    assert env.player_cycle.idx == 0
    env.step(Action.RAISE_POT)
    assert env.current_player.seat == 1
    assert env.player_cycle.idx == 1
    # seat 0 dealer
    env.step(Action.RAISE_POT)  # seat 1 sb
    # assert env.stage == Stage.PREFLOP
    return True

if __name__ == "__main__":
    test_actions()
