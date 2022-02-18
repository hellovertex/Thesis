import re


class ExampleEpisode:
    ep_without_ante = """
  PokerStars Hand #232991295395:  Hold'em No Limit ($1/$2 USD) - 2022/01/16 10:04:49 ET
  Table 'Atalante' 6-max Seat #4 is the button
  Seat 1: jimjames32 ($86 in chips)
  Seat 2: HHnguyen15 ($95.65 in chips)
  Seat 4: kjs609 ($197 in chips)
  jimjames32: posts small blind $1
  HHnguyen15: posts big blind $2
  *** HOLE CARDS ***
  kjs609: raises $2 to $4
  jimjames32: calls $3
  HHnguyen15: calls $2
  *** FLOP *** [Jh 4s 3h]
  jimjames32: checks
  HHnguyen15: checks
  kjs609: bets $5.70
  jimjames32: folds
  HHnguyen15: calls $5.70
  *** TURN *** [Jh 4s 3h] [5d]
  HHnguyen15: checks
  kjs609: bets $11.12
  HHnguyen15: calls $11.12
  *** RIVER *** [Jh 4s 3h 5d] [5h]
  HHnguyen15: checks
  kjs609: bets $43.64
  HHnguyen15: folds
  Uncalled bet ($43.64) returned to kjs609
  kjs609 collected $43.64 from pot
  kjs609: doesn't show hand
  *** SUMMARY ***
  Total pot $45.64 | Rake $2
  Board [Jh 4s 3h 5d 5h]
  Seat 1: jimjames32 (small blind) folded on the Flop
  Seat 2: HHnguyen15 (big blind) folded on the River
  Seat 4: kjs609 (button) collected ($43.64)
  """
    ep_with_ante = """PokerStars Hand #233174710941:  Hold'em No Limit ($0.01/$0.02 USD) - 2022/01/15 19:52:36 ET
Table 'Aaltje' 9-max Seat #2 is the button
Seat 1: imarcu932 ($0.13 in chips)
Seat 2: Vadimirfenix ($2.46 in chips)
Seat 3: gapparov93 ($1.85 in chips)
gapparov93: posts small blind $0.01
imarcu932: posts big blind $0.02
imarcu932: posts the ante $0.01
Vadimirfenix: posts the ante $0.01
gapparov93: posts the ante $0.01
*** HOLE CARDS ***
Vadimirfenix: calls $0.02
gapparov93: calls $0.01
imarcu932: raises $0.10 to $0.12 and is all-in
Vadimirfenix: calls $0.10
gapparov93: folds
*** FLOP *** [2s 5d Ad]
*** TURN *** [2s 5d Ad] [Ts]
*** RIVER *** [2s 5d Ad Ts] [Qd]
*** SHOW DOWN ***
imarcu932: shows [Kc 3c] (high card Ace)
Vadimirfenix: shows [8d Td] (a flush, Ace high)
Vadimirfenix collected $0.28 from pot
*** SUMMARY ***
Total pot $0.29 | Rake $0.01
Board [2s 5d Ad Ts Qd]
Seat 1: imarcu932 (big blind) showed [Kc 3c] and lost with high card Ace
Seat 2: Vadimirfenix (button) showed [8d Td] and won ($0.28) with a flush, Ace high
Seat 3: gapparov93 (small blind) folded before Flop"""


def test_get_ante():
    ante = 0.0
    # todo parse ante if exists
    pattern = re.compile(r'.*? posts the ante ([$€￡]\d+.?\d*)\n')
    res_ante = pattern.findall(ExampleEpisode.ep_with_ante)
    # print(f'res  with ante = {res_ante}'):
    # res_ante = ['$0.01', '$0.01', '$0.01']
    assert res_ante[0]

    res_no_ante = pattern.findall(ExampleEpisode.ep_without_ante)
    # print(f'res without ante = {res_no_ante}')
    # res_no_ante = []
    assert not res_no_ante
