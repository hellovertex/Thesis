# todo: augment state
# todo: optional, change FOLD replacements to be less aggressive
# todo from action-tuple and observation, construct discrete action (i.e. discretize raise amount)
"""
  # todo: check how obs is normalized to avoid small floats

  # *** Observation Augmentation *** #
  # raise vs bet: raise only in preflop stage, bet after preflop
  actions_per_stage = self._init_player_actions(player_info)

  for stage, actions in episode.actions_total.items():
      for action in actions:
          # noinspection PyTypeChecker
          actions_per_stage[action.player_name][stage].append((action.action_type, action.raise_amount))

  # todo: augment inside env wrapper
  # --- Append last 8 moves per player --- #
  # --- Append all players hands --- #
      """
from PokerEnv.PokerRL.game._.EnvWrapperBuilderBase import EnvWrapperBuilderBase
from PokerEnv.PokerRL.game._.wrappers._Wrapper import Wrapper


class AugmentObservationWrapper(Wrapper):

    def __init__(self, env, env_bldr_that_built_me):
        super().__init__(env=env, env_bldr_that_built_me=env_bldr_that_built_me)
        self._list_of_obs_this_episode = None

    def _reset_state(self):
        self._list_of_obs_this_episode = []

    def _pushback(self, env_obs):
        self._list_of_obs_this_episode.append(env_obs)

    def set_to_public_tree_node_state(self, node):
        raise NotImplementedError

    def print_obs(self, wrapped_obs):
        raise NotImplementedError

    def get_current_obs(self, env_obs):
        return env_obs

    @property
    def current_player(self):
        return self.env.current_player


class AugmentedEnvBuilder(EnvWrapperBuilderBase):
    WRAPPER_CLS = AugmentObservationWrapper

    def __init__(self, env_cls, env_args):
        super().__init__(env_cls=env_cls, env_args=env_args)

    def _get_num_public_observation_features(self):
        return super()._get_num_public_observation_features()

    def _get_num_private_observation_features(self):
        return super()._get_num_private_observation_features()

    def _get_obs_parts_idxs(self):
        return super()._get_obs_parts_idxs()
