""" This module will
 - read .txt files inside ./data/
 - parse them to create corresponding environment states.
 - The states are encoded as in neuron_poker.gym_env.env.HoldemTable  """
from time import time
import re


def get_button(s: str):
    ptn_button = re.compile(r"Seat #\d is the button")
    # ptn_button2 = re.compile(r"Seat #(\d) is the button")
    # button2 = int(ptn_button2.findall(s)[0])
    button = ptn_button.findall(s)[0][6]
    return int(button)  # 1-indexed


def get_player_stacks(s: str):
    pattern = re.compile(r"(Seat \d): ([a-zA-Z0-9]+) \(([$€]\d+.?\d*)")
    return pattern.findall(s)


def get_blinds(s: str):
    pattern = re.compile(r"([a-zA-Z0-9]+): posts (small blind|big blind) ([$€]\d+.?\d*)")
    return pattern.findall(s)


def main():
    t0 = time()
    with open('../data/Atalante-1-2-USD-NoLimitHoldem-PokerStarsPA-1-16-2022.txt', 'r') as f:
        content = f.read()
        print(f'loading took {time() - t0} seconds...')
        content = re.split(r'PokerStars Hand #', content)

        for c in content:
            if c == '': continue  # pylint: disable=multiple-statements
            print(c, type(c))
            btn = get_button(c)
            player_stacks = get_player_stacks(c)
            blinds = get_blinds(c)
            break


if __name__ == "__main__":
    """
    Mobster365: folds
    sziget37: folds
    Kraliya: raises $0.10 to $0.20
    sts1981: folds
    alaver24: raises $0.50 to $0.70
    milkimen: folds
    BUICKBOY2: calls $0.65
    PlayLaughGro: folds
    Kraliya: raises $0.50 to $1.20
    alaver24: calls $0.50
    BUICKBOY2: raises $1.53 to $2.73 and is all-in
    Kraliya: raises $1.53 to $4.26
    alaver24: folds
    Uncalled bet ($1.53) returned to Kraliya
    ... showdown yadayadayada
    """
    main()
