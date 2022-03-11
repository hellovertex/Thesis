from __future__ import annotations

import argparse
from functools import partial
from os import listdir, mkdir
from os.path import isfile, join, abspath, basename, exists
from torch.utils.data import ConcatDataset, DataLoader

import numpy as np
import pandas as pd
import torch

DATA_DIR = '../../../../data/'


class SingleTxtFileDataset(torch.utils.data.Dataset):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._data: pd.DataFrame | None = None  # todo sort
        self._has_been_consumed = False

    def load_file(self) -> pd.DataFrame:
        # loading
        df = pd.read_csv(self.file_path, sep=",")

        # preprocesing
        fn_to_numeric = partial(pd.to_numeric, errors="coerce")
        self._has_been_consumed = False
        return df.apply(fn_to_numeric).dropna().astype(dtype=np.float32)

    def __getitem__(self, idx):
        if self._data is None:
            self._data = self.load_file()
        y = self._data.loc[idx].pop('label')
        return self._data.loc[idx], y

    def __len__(self):
        if self._data is None:
            return 0
        return len(self._data)



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

    # list_of_datasets = [SingleTxtFileDataset(train_file) for train_file in train_files]
    # multiple_file_dataset = ConcatDataset(list_of_datasets)
    # loader = DataLoader(dataset=multiple_file_dataset, batch_size=1)
    # print(len(list_of_datasets))
    # print(len(multiple_file_dataset))
    # for batch_ndx, sample in enumerate(loader):
    #     print('batch_ndx:', batch_ndx, 'sample:', sample)
    #     break

    # write to disk [args.preprocessed_dir]
    preprocessed_dir = args.preprocessed_dir
    if preprocessed_dir is None:
        preprocessed_dir = DATA_DIR + 'train_data/0.25_0.50/preprocessed/'

    def one_time_fix():
        for file_path in train_files:
            df = pd.read_csv(file_path, sep=",")
            fn_to_numeric = partial(pd.to_numeric, errors="coerce")
            df = df.apply(fn_to_numeric).dropna().astype(dtype=np.float32)
            out_file = abspath(preprocessed_dir + basename(file_path))
            if not exists(preprocessed_dir):
                mkdir(preprocessed_dir)
            df.to_csv(out_file, index_label='label', mode='w+')
            print(f'written to {out_file}')
    one_time_fix()

    # check kaggle for visualization tips
    # check machine learning from multiple text files
    # setup mlflow locally
    # setup mlflow in azure asai runs locally