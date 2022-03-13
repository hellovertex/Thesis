from __future__ import annotations

import argparse
from functools import partial
from os import listdir, mkdir
from os.path import isfile, join, abspath, basename, exists

import psutil
from torch.utils.data import ConcatDataset, DataLoader
from typing import Tuple

import numpy as np
import pandas as pd
import torch

import train

DATA_DIR = '../../../../data/'
BATCH_SIZE = 64


class SingleTxtFileDataset(torch.utils.data.Dataset):
    def __init__(self, file_path: str):
        print(f"Initializing file {file_path}")
        self.file_path = file_path
        self._data: torch.Tensor | None = None  # todo sort
        self._labels: torch.Tensor | None = None
        self._len = self._get_len()

    def _get_len(self):
        """Get line count of large files cheaply"""
        with open(self.file_path, 'rb') as f:
            lines = 0
            buf_size = 1024 * 1024
            read_f = f.raw.read if hasattr(f, 'raw') and hasattr(f.raw, 'read') else f.read

            buf = read_f(buf_size)
            while buf:
                lines += buf.count(b'\n')
                buf = read_f(buf_size)

        return lines

    def load_file(self):
        # loading
        df = pd.read_csv(self.file_path, sep=",")

        # preprocesing
        fn_to_numeric = partial(pd.to_numeric, errors="coerce")
        df = df.apply(fn_to_numeric).dropna().astype(dtype=np.float32)
        labels = df.pop('label')
        assert len(df.index) > 0
        self._data = torch.tensor(df.values)
        self._labels = torch.tensor(labels.values)

    def __getitem__(self, idx):
        if self._data is None:
            assert self._labels is None
            self.load_file()

        return self._data[idx], self._labels[idx]

    def __len__(self):
        return self._len


def get_dataloaders(train_dir):
    """Makes torch dataloaders by reading training directory files.
    1: Load training data files
    2: Create train, val, test splits
    3: Make datasets for each split
    4: Return dataloaders for each dataset
    """
    # get list of .txt-files inside train_dir
    train_dir_files = [join(train_dir, f) for f in listdir(train_dir) if isfile(join(train_dir, f))]

    # by convention, any .txt files inside this folder
    # that do not have .meta in their name, contain training data
    train_dir_files = [abspath(f) for f in train_dir_files if ".meta" not in f]

    print(train_dir_files)
    print(f'{len(train_dir_files)} train files loaded')

    # splits
    total_count = len(train_dir_files)
    train_count = int(0.7 * total_count)
    valid_count = int(0.2 * total_count)
    test_count = total_count - train_count - valid_count

    # splits filepaths
    train_files = train_dir_files[:train_count]
    valid_files = train_dir_files[train_count:train_count + valid_count]
    test_files = train_dir_files[-test_count:]

    # splits datasets
    train_dataset = ConcatDataset(
        [SingleTxtFileDataset(train_file) for train_file in train_files])
    valid_dataset = ConcatDataset(
        [SingleTxtFileDataset(train_file) for train_file in valid_files])
    test_dataset = ConcatDataset(
        [SingleTxtFileDataset(train_file) for train_file in test_files])

    print(f'Number of files loaded = {len(train_files) + len(test_files) + len(valid_files)}')
    print(f'For a total number of {len(test_dataset) + len(train_dataset) + len(valid_dataset)} examples')

    train_dataset_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=psutil.cpu_count(logical=False)
    )
    valid_dataset_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=psutil.cpu_count(logical=False)
    )
    test_dataset_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=psutil.cpu_count(logical=False)
    )
    dataloaders = {
        "train": train_dataset_loader,
        "val": valid_dataset_loader,
        "test": test_dataset_loader,
    }

    return dataloaders


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
        train_dir = DATA_DIR + 'train_data/0.25_0.50/preprocessed'

    dataloaders = get_dataloaders(train_dir)
    train_loader = dataloaders['train']
    test_loader = dataloaders['test']
    train.driver(train_loader, test_loader)


# def one_time_fix():
#     for file_path in train_files:
#         df = pd.read_csv(file_path, sep=",")
#         fn_to_numeric = partial(pd.to_numeric, errors="coerce")
#         df = df.apply(fn_to_numeric).dropna().astype(dtype=np.float32)
#         out_file = abspath(preprocessed_dir + basename(file_path))
#         if not exists(preprocessed_dir):
#             mkdir(preprocessed_dir)
#         df.to_csv(out_file, index_label='label', mode='w+')
#         print(f'written to {out_file}')
# one_time_fix()
