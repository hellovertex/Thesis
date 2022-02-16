from txt_generator import CsvGenerator
from txt_parser import TxtParser
from PokerRL_encoder import RLStateEncoder
from PokerRL_wrapper import AugmentObservationWrapper
import os
import glob

DATA_DIR = '../../../data/'


def main(filenames: list):
    parser = TxtParser()
    # use AugmentedEnvBuilder to get augmented observations encodings
    encoder = RLStateEncoder(env_wrapper_cls=AugmentObservationWrapper)

    with CsvGenerator(data_dir=DATA_DIR,
                      out_dir=os.path.join(DATA_DIR + 'train_data'),
                      parser=parser,
                      encoder=encoder,
                      out_filename='6MAX_0.25USD_0.50USD_Pokerstars_eu.txt',
                      write_azure=False) as generator:

        for i, filename in enumerate(filenames):
            # out_subdir = generator.get_out_subdir(filename)
            print(f"Encoding training data from file {i}/{len(filenames)}:"
                  f"{filename}")

            generator.generate_from_file(filename, out_subdir='0.25_0.50')
            break


if __name__ == '__main__':
    # UNZIPPED_DATA_DIR = DATA_DIR + '/0.25-0.50'
    UNZIPPED_DATA_DIR = DATA_DIR + '/examples_unprocessed'
    filenames_recursively = glob.glob(UNZIPPED_DATA_DIR.__str__() + '/**/*.txt', recursive=True)

    # os.walk here to generate list of files
    main(filenames_recursively)
