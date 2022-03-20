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
from azureml.core import Workspace

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
    ws = Workspace.from_config()
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

    with mlflow.start_run() as run:
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
    return run


if __name__ == "__main__":
    train_dir = DATA_DIR + '03_preprocessed/0.25-0.50/preprocessed'

    dataloaders = get_dataloaders(train_dir)
    train_loader = dataloaders['train']
    test_loader = dataloaders['test']

    mlflow_run = run(train_loader, test_loader)
