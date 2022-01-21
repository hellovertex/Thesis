""" This module will
 - read .txt files inside ./data/
 - parse them to create corresponding environment states.
 - The states are encoded as in neuron_poker.gym_env.env.HoldemTable  """
from time import time
import re


def get_button(s: str):
    ptn_button = re.compile(r"Seat #\d is the button")
    button = ptn_button.findall(s)[0][6]
    return int(button)  # 1-indexed


def get_players(s: str):
    pattern = re.compile(r"Seat \d: [a-zA-Z0-9]+")
    players = pattern.findall(s)
    return [player[8:] for player in players]


def get_player_stacks(s: str):
    return ""


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
            player_names = get_players(c)

            break


if __name__ == "__main__":
    main()
