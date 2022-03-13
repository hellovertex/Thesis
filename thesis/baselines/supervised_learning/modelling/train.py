from __future__ import annotations

import argparse
import tempfile
from functools import partial
from os import listdir
from os.path import isfile, join, abspath

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import psutil
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import ConcatDataset

from model import Net, train, test

DATA_DIR = '../../../../data/'
BATCH_SIZE = 64


class Args(object):
    pass


# Training settings
args = Args()
setattr(args, 'batch_size', 64)
setattr(args, 'test_batch_size', 1000)
setattr(args, 'epochs', 3)
setattr(args, 'lr', 1e-6)
setattr(args, 'momentum', 0.5)
setattr(args, 'use_cuda', torch.cuda.is_available())
setattr(args, 'seed', 1)
setattr(args, 'log_interval', 10)
setattr(args, 'log_artifacts_interval', 10)
setattr(args, 'save_model', True)
setattr(args, 'output_dir', tempfile.mkdtemp())
use_cuda = not args.use_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


class SingleTxtFileDataset(torch.utils.data.Dataset):
    def __init__(self, file_path: str):
        print(f"Initializing file {file_path}")
        self.file_path = file_path
        self._data: torch.Tensor | None = None
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

        # preprocessing
        fn_to_numeric = partial(pd.to_numeric, errors="coerce")
        df = df.apply(fn_to_numeric).dropna()
        labels = None
        try:
            # todo remove this when we do not have
            # todo two label columns by accident anymore
            labels = df.pop('label.1')
        except KeyError:
            labels = df.pop('label')
        assert len(df.index) > 0
        self._data = torch.tensor(df.values, dtype=torch.float32)
        self._labels = torch.tensor(labels.values, dtype=torch.long)

    def __getitem__(self, idx):
        if self._data is None:
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
    train_dir_files = [abspath(f) for f in train_dir_files if ".meta" not in f][:5]

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


def run(train_loader, test_loader):
    # warnings.filterwarnings("ignore")
    # # Dependencies for deploying the model
    # pytorch_index = "https://download.pytorch.org/whl/"
    # pytorch_version = "cpu/torch-1.1.0-cp36-cp36m-linux_x86_64.whl"
    # deps = [
    #     "cloudpickle=={}".format(cloudpickle.__version__),
    #     pytorch_index + pytorch_version,
    # ]

    with mlflow.start_run():
        try:
            # Since the model was logged as an artifact, it can be loaded to make predictions
            model = mlflow.pytorch.load_model(mlflow.get_artifact_uri("pytorch-model"))
        except Exception as e:
            print(e)
            input_dim = output_dim = hidden_dim = None
            for data, label in train_loader:
                input_dim = data.shape[1]
                output_dim = label.shape[0]
                hidden_dim = [512, 512]
                break
            model = Net(input_dim, output_dim, hidden_dim).to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-6)
        # Create a SummaryWriter to write TensorBoard events locally

        writer = SummaryWriter(args.output_dir)
        print("Writing TensorBoard events locally to %s\n" % args.output_dir)

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch, writer)
            test(epoch, args, model, test_loader, train_loader, writer)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Provide model pipeline with necessary arguments: "
                    "- path to training data "
                    "-whatever else comes to my mind later")
    argparser.add_argument('-t', '--train_dir',
                           help='abs or rel path to .txt files with raw training samples.')
    argparser.add_argument('-p', '--preprocessed_dir',
                           help='abs or rel path to .txt files with preprocessed training samples.')

    cmd_args, _ = argparser.parse_known_args()
    train_dir = cmd_args.train_dir
    if cmd_args.train_dir is None:
        train_dir = DATA_DIR + 'train_data/0.25_0.50/preprocessed'

    dataloaders = get_dataloaders(train_dir)
    train_loader = dataloaders['train']
    test_loader = dataloaders['test']

    run(train_loader, test_loader)
