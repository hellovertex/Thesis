from PokerRL_generator import PokerRLGenerator
from txt_parser import TxtParser
from PokerRL_encoder import RLStateEncoder
from PokerRL_wrapper import AugmentObservationWrapper

DATA_DIR = '../../../data/'


def main(filename):
    parser = TxtParser()
    # use AugmentedEnvBuilder to get augmented observations encodings
    encoder = RLStateEncoder(env_wrapper_cls=AugmentObservationWrapper)

    generator = PokerRLGenerator(data_dir=DATA_DIR,
                          out_dir=DATA_DIR + '/train_data/',
                          parser=parser,
                          encoder=encoder,
                          out_filename='6MAX_0.25USD_0.50USD_Pokerstars_eu.txt')
    generator.generate_from_file(filename, out_subdir='0.25_0.50')


if __name__ == '__main__':
    FILENAME = 'Atalante-1-2-USD-NoLimitHoldem-PokerStarsPA-1-16-2022.txt'
    main(FILENAME)
