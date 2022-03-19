from __future__ import annotations

import argparse
import os
import tempfile

import mlflow
import mlflow.pytorch
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from dataset import get_dataloaders
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
setattr(args, 'output_dir', "artifacts")
use_cuda = not args.use_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


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
            model = Net(input_dim, output_dim, hidden_dim)
        if args.use_cuda:
            model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=1e-6)
        # Create a SummaryWriter to write TensorBoard events locally

        writer = SummaryWriter(args.output_dir)
        print("Writing TensorBoard events locally to %s\n" % args.output_dir)
        print(
            "\nLaunch TensorBoard with:\n\ntensorboard --logdir=%s"
            % os.path.join(mlflow.get_artifact_uri(), "events")
        )
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
        train_dir = DATA_DIR + '03_preprocessed/0.25-0.50/preprocessed'

    dataloaders = get_dataloaders(train_dir)
    train_loader = dataloaders['train']
    test_loader = dataloaders['test']

    run(train_loader, test_loader)
