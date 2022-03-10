import os
import glob
import pathlib
import argparse

from .txt_generator import CsvGenerator
from .txt_parser import TxtParser
from thesis.baselines.supervised_learning.data.PokerRL_encoder import RLStateEncoder
from thesis.baselines.supervised_learning.data.PokerRL_wrapper import AugmentObservationWrapper

DATA_DIR = '../../../../data/'


def main(filenames: list):
    parser = TxtParser()
    # use AugmentedEnvBuilder to get augmented observations encodings
    encoder = RLStateEncoder(env_wrapper_cls=AugmentObservationWrapper)
    logfile = "log.txt"
    with CsvGenerator(data_dir=DATA_DIR,
                      out_dir=os.path.join(DATA_DIR + 'train_data'),
                      parser=parser,
                      encoder=encoder,
                      out_filename='6MAX_0.25USD_0.50USD_Pokerstars_eu.txt',
                      write_azure=False,
                      logfile=logfile) as generator:

        for i, filename in enumerate(filenames):
            # out_subdir = generator.get_out_subdir(filename)
            # print(f"Encoding training data from file {i}/{len(filenames)}:"
            #       f"{filename}")
            skip_file = False
            # skip already encoded files
            with open(DATA_DIR + logfile, "a+") as f:
                files_written = f.readlines().__reversed__()
                for fw in files_written:
                    if filename in fw:
                        print(f"Skipping file {filename} because it has already been encoded and written to disk...")
                        skip_file=True
                        break
            if not skip_file:
                generator.generate_from_file(filename, out_subdir='0.25_0.50')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description="Use to pass the directories for source and target text files.")

    argparser.add_argument('-s', '--source_dir', help='abs or rel path to .txt files')
    argparser.add_argument('-t', '--target_dir', help='path where generated training .txt files are stored.')
    args, _ = argparser.parse_known_args()
    if args.source_dir is None:
        # UNZIPPED_DATA_DIR = DATA_DIR + '/0.25-0.50'
        # data/0.25-0.50/BulkHands-14686/unzipped
        UNZIPPED_DATA_DIR = DATA_DIR + '0.25-0.50/unzipped'
        # UNZIPPED_DATA_DIR = "/home/cawa/Documents/github.com/hellovertex/Thesis/data/6Max_Regular_0.25-0.50_PokerStars_eu/unzipped"
        # print(pathlib.Path(UNZIPPED_DATA_DIR).resolve())
        filenames_recursively = glob.glob(UNZIPPED_DATA_DIR.__str__() + '/**/*.txt', recursive=True)
    else:
        filenames_recursively = glob.glob(args.source_dir.__str__() + '/**/*.txt', recursive=True)
    # filenames_recursively = [DATA_DIR + "AAA.txt"]
    # filenames_recursively = [DATA_DIR + "Aaltje-0.01-0.02-USD-NoLimitHoldem-PokerStars-1-16-2022.txt"]
    print(filenames_recursively)

    # generate list of files
    main(filenames_recursively)
