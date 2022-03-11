import argparse
import glob
import pandas as pd
import numpy as np
import torch

from functools import partial
from os import listdir
from os.path import isfile, join, abspath

DATA_DIR = '../../../../data/'

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Provide model pipeline with necessary arguments: "
                    "- path to training data "
                    "-whatever else comes to my mind later")
    argparser.add_argument('-t', '--train_dir',
                           help='abs or rel path to .txt files with raw training samples.')
    argparser.add_argument('-p', '--preprocessed_dir',
                           help='abs or rel path to .txt files with preprocessed training samples.')

    args, _ = argparser.parse_known_args()
    train_dir = args.train_dir
    if args.train_dir is None:
        train_dir = DATA_DIR + 'train_data/0.25_0.50/'

    # get list of .txt-files inside train_dir
    train_dir_files = [join(train_dir, f) for f in listdir(train_dir) if isfile(join(train_dir, f))]

    # by convention, any .txt files inside this folder
    # that do not have .meta in their name, contain training data
    train_files = [abspath(f) for f in train_dir_files if ".meta" not in f]

    print(train_dir_files)
    print(train_files)

    # preprocessing
    df = pd.read_csv(train_files[0], sep=",")
    from functools import partial

    fn = partial(pd.to_numeric, errors="coerce")
    df = df.apply(fn).dropna().astype(dtype=np.float32)
    print(df.memory_usage(index=True, deep=True).sum())
    print(df.head())

    ys = df.pop('label')
    print(df.head())


    # write to disk [args.preprocessed_dir]
    # check kaggle for visualization tips
    # check machine learning from multiple text files
    def load_file(f):
        return None


    class MyDataset(torch.utils.Dataset):
        def __init__(self):
            self.data_files = listdir('data_dir')  # todo sort

        def __getindex__(self, idx):
            return load_file(self.data_files[idx])

        def __len__(self):
            return len(self.data_files)
