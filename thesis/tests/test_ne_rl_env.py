from thesis import ne_rl_env


def test_make_env():
    # Create NoLimitTexasHoldEm game with full 52 card deck
    num_players = 6
    env = ne_rl_env.make("NLHE-Full", num_players=6)
    assert env.game.num_players() == num_players
