from txt_generator import CsvGenerator
from txt_parser import TxtParser
from PokerRL_encoder import RLStateEncoder
from PokerRL_wrapper import AugmentObservationWrapper
import os
import glob
import pathlib

DATA_DIR = '../../../data/'


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
            with open(DATA_DIR + logfile, "r") as f:
                files_written = f.readlines().__reversed__()
                for fw in files_written:
                    if filename in fw:
                        print(f"Skipping file {filename} because it has already been encoded and written to disk...")
                        skip_file=True
                        break
            if not skip_file:
                generator.generate_from_file(filename, out_subdir='0.25_0.50')


if __name__ == '__main__':
    # UNZIPPED_DATA_DIR = DATA_DIR + '/0.25-0.50'
    # data/0.25-0.50/BulkHands-14686/unzipped
    UNZIPPED_DATA_DIR = DATA_DIR + '0.25-0.50/BulkHands-14686/unzipped'
    print(pathlib.Path(UNZIPPED_DATA_DIR).resolve())
    filenames_recursively = glob.glob(UNZIPPED_DATA_DIR.__str__() + '/**/*.txt', recursive=True)
    # filenames_recursively = [DATA_DIR + "AAA.txt"]
    # print(filenames_recursively)
    # os.walk here to generate list of files
    main(filenames_recursively)
