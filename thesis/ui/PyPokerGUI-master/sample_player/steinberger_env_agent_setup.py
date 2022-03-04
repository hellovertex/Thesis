from pypokerengine.players import BasePokerPlayer
from PokerRL.game.games import NoLimitHoldem
from PokerRL_wrapper import AugmentObservationWrapper


class SteinbergerEnvAgent(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"

    _default_deck = None
    _default_hole_card = (None, None)
    _default_board = None

    def __init__(self, env_wrapper_cls=None):
        self._player_num = None
        self._ante = None
        self._sb_amt = None
        self._bb_amt = None
        self._starting_stack_sizes: list = []
        self._seats: list = []
        self._env = None
        self._env_wrapper_class = AugmentObservationWrapper if env_wrapper_cls is None else env_wrapper_cls
        self._curr_obs = None
        self.net = None

    def reset(self):
        pass

    def _update_cards_state(self, cards_state_dict):
        """Synchronizes underlying environment board cards and whole cards.
        When the gui notifies of an update, e.g. a drawn card, this method calls
        the environemnts load_cards_state_dict function.
        Args:
            cards_state_dict: {'deck': [...], 'board': [...], 'hand': [...]}
        """

    def _format_action(self, action):
        pass
    
    # using self._curr_obs: current observation from self._env to generate action
    # smooth out self.net(obs) = action, if its not contained in valid actions
    def declare_action(self, valid_actions, hole_card, round_state):
        # action = self.net(self._curr_obs)
        # action = self._format_action(action)
        # valid_actions format => [raise_action_info, call_action_info, fold_action_info]
        call_action_info = valid_actions[1]
        action, amount = call_action_info["action"], call_action_info["amount"]
        return action, amount  # action returned here is sent to the poker engine

    # init env
    def receive_game_start_message(self, game_info):
        """ Initialize simulation environment. It will be used to generate observations for the agent.
        game_info: {
              'player_num': 3,
              'rule': {
                'ante': 5,
                'blind_structure': {  # can be empty
                  5 : { "ante": 10, "small_blind": 20 },
                  7 : { "ante": 15, "small_blind": 30 }
                },
                'max_round': 10,
                'initial_stack': 100,
                'small_blind_amount': 10
              },
              'seats': [
                {'stack': 100, 'state': 'participating', 'name': 'p1', 'uuid': 'ftwdqkystzsqwjrzvludgi'},
                {'stack': 100, 'state': 'participating', 'name': 'p2', 'uuid': 'bbiuvgalrglojvmgggydyt'},
                {'stack': 100, 'state': 'participating', 'name': 'p3', 'uuid': 'zkbpehnazembrxrihzxnmt'}
              ]
            }
        """
        self._player_num = game_info['player_num']
        self._ante = game_info['rule']['ante']
        self._sb_amt = game_info['rule']['small_blind_amount']
        self._bb_amt = self._sb_amt * 2  # todo check if *2 is correct
        initial_stack_size = game_info['initial_stack']
        self._starting_stack_sizes = [initial_stack_size for _ in range(self._player_num)]
        self._seats = [{'stack': initial_stack_size} for _ in range(self._player_num)]
        cards_state_dict = {'deck': self._default_deck,
                            'board': self._default_board,
                            'hand': [self._default_hole_cards for _ in range(self._player_num)]}
        # make args for env
        args = NoLimitHoldem.ARGS_CLS(n_seats=self._player_num,
                                      starting_stack_sizes_list=self._starting_stack_sizes_list)
        # initialize wrapped env instance
        env = NoLimitHoldem(is_evaluating=True,
                            env_args=args,
                            lut_holder=NoLimitHoldem.get_lut_holder())
        self._env = self.env_wrapper_cls(env)
        self._env.SMALL_BLIND = self._sb_amt
        self._env.BIG_BLIND = self._bb_amt
        self._env.ANTE = self._ante
        self._curr_obs, _, done, _ = self._env.reset(config={'deck_state_dict': cards_state_dict})

    # load cards
    def receive_round_start_message(self, round_count, hole_card, seats):
        """Store round start data.
        Args:
            round_count: int, number of ...
            hole_card: e.g. ['C2', 'HQ']
            seats: e.g. [
                      {'stack': 135, 'state': 'participating', 'name': 'p1', 'uuid': 'ftwdqkystzsqwjrzvludgi'},
                      {'stack': 80, 'state': 'participating', 'name': 'p2', 'uuid': 'bbiuvgalrglojvmgggydyt'},
                      {'stack': 40, 'state': 'participating', 'name': 'p3', 'uuid': 'zkbpehnazembrxrihzxnmt'}
                  ]
        """
        # todo what is round_count
        # todo call _update_cards_state(...)
        pass

    # load cards
    def receive_street_start_message(self, street: str, round_state: dict):
        """Stores stree start data.
        Args:
            street: e.g. 'preflop'
            round_state: e.g. {
              'round_count': 1,
              'dealer_btn': 0,
              'small_blind_pos': 1,
              'big_blind_pos': 2,
              'next_player': 0,
              'small_blind_amount': 10,
              'street': 'preflop',
              'community_card': [],
              'seats': [
                {'stack': 95, 'state': 'participating', 'name': 'p1', 'uuid': 'ftwdqkystzsqwjrzvludgi'},
                {'stack': 85, 'state': 'participating', 'name': 'p2', 'uuid': 'bbiuvgalrglojvmgggydyt'},
                {'stack': 75, 'state': 'participating', 'name': 'p3', 'uuid': 'zkbpehnazembrxrihzxnmt'}
               ],
              'pot': {'main': {'amount': 45}, 'side': [] },
              'action_histories': {
                'preflop': [
                  {'action': 'ANTE', 'amount': 5, 'uuid': 'bbiuvgalrglojvmgggydyt'},
                  {'action': 'ANTE', 'amount': 5, 'uuid': 'zkbpehnazembrxrihzxnmt'},
                  {'action': 'ANTE', 'amount': 5, 'uuid': 'ftwdqkystzsqwjrzvludgi'},
                  {'action': 'SMALLBLIND', 'amount': 10, 'add_amount': 10, 'uuid': 'bbiuvgalrglojvmgggydyt'},
                  {'action': 'BIGBLIND', 'amount': 20, 'add_amount': 10, 'uuid': 'zkbpehnazembrxrihzxnmt'}
                ]}}
        """
        pass

    # step env
    def receive_game_update_message(self, action, round_state):
        """Store game update data.
        Args:
            action: {
                      'player_uuid': 'ftwdqkystzsqwjrzvludgi',
                      'action': 'raise',
                      'amount': 30
                    }
            round_state: {
              'dealer_btn': 1,
              'big_blind_pos': 0,
              'round_count': 2,
              'small_blind_pos': 2,
              'next_player': 0,
              'small_blind_amount': 10,
              'action_histories': {
                'turn': [
                  {'action': 'CALL', 'amount': 0, 'uuid': 'ftwdqkystzsqwjrzvludgi', 'paid': 0},
                  {'action': 'RAISE', 'amount': 20, 'add_amount': 20, 'paid': 20, 'uuid': 'bbiuvgalrglojvmgggydyt'},
                  {'action': 'CALL', 'amount': 20, 'uuid': 'ftwdqkystzsqwjrzvludgi', 'paid': 20}
                ],
                'preflop': [
                  {'action': 'ANTE', 'amount': 5, 'uuid': 'zkbpehnazembrxrihzxnmt'},
                  {'action': 'ANTE', 'amount': 5, 'uuid': 'ftwdqkystzsqwjrzvludgi'},
                  {'action': 'ANTE', 'amount': 5, 'uuid': 'bbiuvgalrglojvmgggydyt'},
                  {'action': 'SMALLBLIND', 'amount': 10, 'add_amount': 10, 'uuid': 'zkbpehnazembrxrihzxnmt'},
                  {'action': 'BIGBLIND', 'amount': 20, 'add_amount': 10, 'uuid': 'ftwdqkystzsqwjrzvludgi'},
                  {'action': 'CALL', 'amount': 20, 'uuid': 'bbiuvgalrglojvmgggydyt', 'paid': 20},
                  {'action': 'RAISE', 'amount': 30, 'add_amount': 10, 'paid': 20, 'uuid': 'zkbpehnazembrxrihzxnmt'},
                  {'action': 'CALL', 'amount': 30, 'uuid': 'ftwdqkystzsqwjrzvludgi', 'paid': 10},
                  {'action': 'CALL', 'amount': 30, 'uuid': 'bbiuvgalrglojvmgggydyt', 'paid': 10}
                ],
                'river': [],
                'flop': [
                  {'action': 'CALL', 'amount': 0, 'uuid': 'zkbpehnazembrxrihzxnmt', 'paid': 0},
                  {'action': 'RAISE', 'amount': 30, 'add_amount': 30, 'paid': 30, 'uuid': 'ftwdqkystzsqwjrzvludgi'},
                  {'action': 'CALL', 'amount': 30, 'uuid': 'bbiuvgalrglojvmgggydyt', 'paid': 30},
                  {'action': 'CALL', 'amount': 20, 'uuid': 'zkbpehnazembrxrihzxnmt', 'paid': 20}
                ]
              },
              'street': 'showdown',
              'seats': [
                {'stack': 300, 'state': 'participating', 'name': 'p1', 'uuid': 'ftwdqkystzsqwjrzvludgi'},
                {'stack': 0, 'state': 'allin', 'name': 'p2', 'uuid': 'bbiuvgalrglojvmgggydyt'},
                {'stack': 0, 'state': 'allin', 'name': 'p3', 'uuid': 'zkbpehnazembrxrihzxnmt'}
              ],
              'community_card': ['DJ', 'H6', 'S6', 'H5', 'C4'],
              'pot': {
                'main': {'amount': 165},
                'side': [
                  {'amount': 60, 'eligibles': ['ftwdqkystzsqwjrzvludgi', 'bbiuvgalrglojvmgggydyt'] },
                  {'amount': 0, 'eligibles': ['ftwdqkystzsqwjrzvludgi', 'bbiuvgalrglojvmgggydyt'] }
                ]}}
        """
        # todo step environment
        pass

    # reset agent
    def receive_round_result_message(self, winners, hand_info, round_state):
        """Stores round result data.
        Args:
            winners: [
                {'stack': 300, 'state': 'participating', 'name': 'p1', 'uuid': 'ftwdqkystzsqwjrzvludgi'}
            ]
            hand_info: [
              {
                'uuid': 'ftwdqkystzsqwjrzvludgi',
                'hand': {
                  'hole': {'high': 14, 'low': 13},
                  'hand': {'high': 6, 'strength': 'ONEPAIR', 'low': 0}
                }
              },
              {
                'uuid': 'bbiuvgalrglojvmgggydyt',
                'hand': {
                  'hole': {'high': 12, 'low': 2},
                  'hand': {'high': 6, 'strength': 'ONEPAIR', 'low': 0}
                }
              },
              {
                'uuid': 'zkbpehnazembrxrihzxnmt',
                'hand': {
                  'hole': {'high': 10, 'low': 3},
                  'hand': {'high': 6, 'strength': 'ONEPAIR', 'low': 0}
                }
              }
            ]
            round_state: {
              'dealer_btn': 1,
              'big_blind_pos': 0,
              'round_count': 2,
              'small_blind_pos': 2,
              'next_player': 0,
              'small_blind_amount': 10,
              'action_histories': {
                'turn': [
                  {'action': 'CALL', 'amount': 0, 'uuid': 'ftwdqkystzsqwjrzvludgi', 'paid': 0},
                  {'action': 'RAISE', 'amount': 20, 'add_amount': 20, 'paid': 20, 'uuid': 'bbiuvgalrglojvmgggydyt'},
                  {'action': 'CALL', 'amount': 20, 'uuid': 'ftwdqkystzsqwjrzvludgi', 'paid': 20}
                ],
                'preflop': [
                  {'action': 'ANTE', 'amount': 5, 'uuid': 'zkbpehnazembrxrihzxnmt'},
                  {'action': 'ANTE', 'amount': 5, 'uuid': 'ftwdqkystzsqwjrzvludgi'},
                  {'action': 'ANTE', 'amount': 5, 'uuid': 'bbiuvgalrglojvmgggydyt'},
                  {'action': 'SMALLBLIND', 'amount': 10, 'add_amount': 10, 'uuid': 'zkbpehnazembrxrihzxnmt'},
                  {'action': 'BIGBLIND', 'amount': 20, 'add_amount': 10, 'uuid': 'ftwdqkystzsqwjrzvludgi'},
                  {'action': 'CALL', 'amount': 20, 'uuid': 'bbiuvgalrglojvmgggydyt', 'paid': 20},
                  {'action': 'RAISE', 'amount': 30, 'add_amount': 10, 'paid': 20, 'uuid': 'zkbpehnazembrxrihzxnmt'},
                  {'action': 'CALL', 'amount': 30, 'uuid': 'ftwdqkystzsqwjrzvludgi', 'paid': 10},
                  {'action': 'CALL', 'amount': 30, 'uuid': 'bbiuvgalrglojvmgggydyt', 'paid': 10}
                ],
                'river': [],
                'flop': [
                  {'action': 'CALL', 'amount': 0, 'uuid': 'zkbpehnazembrxrihzxnmt', 'paid': 0},
                  {'action': 'RAISE', 'amount': 30, 'add_amount': 30, 'paid': 30, 'uuid': 'ftwdqkystzsqwjrzvludgi'},
                  {'action': 'CALL', 'amount': 30, 'uuid': 'bbiuvgalrglojvmgggydyt', 'paid': 30},
                  {'action': 'CALL', 'amount': 20, 'uuid': 'zkbpehnazembrxrihzxnmt', 'paid': 20}
                ]
              },
              'street': 'showdown',
              'seats': [
                {'stack': 300, 'state': 'participating', 'name': 'p1', 'uuid': 'ftwdqkystzsqwjrzvludgi'},
                {'stack': 0, 'state': 'allin', 'name': 'p2', 'uuid': 'bbiuvgalrglojvmgggydyt'},
                {'stack': 0, 'state': 'allin', 'name': 'p3', 'uuid': 'zkbpehnazembrxrihzxnmt'}
              ],
              'community_card': ['DJ', 'H6', 'S6', 'H5', 'C4'],
              'pot': {
                'main': {'amount': 165},
                'side': [
                  {'amount': 60, 'eligibles': ['ftwdqkystzsqwjrzvludgi', 'bbiuvgalrglojvmgggydyt'] },
                  {'amount': 0, 'eligibles': ['ftwdqkystzsqwjrzvludgi', 'bbiuvgalrglojvmgggydyt'] }
                ]
              }
            }
            """
        # todo reset agent ?
        pass


def setup_ai():
    return SteinbergerEnvAgent()
