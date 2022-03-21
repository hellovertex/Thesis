# from __future__ import annotations

import argparse
import glob
import io
import os
import tempfile
import zipfile

import mlflow
import mlflow.pytorch
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from dataset import get_dataloaders
from model import Net, train, test
from azureml.core import Workspace

# DATA_DIR = '../../../../data/'
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


def extract(filename, out_dir):
    z = zipfile.ZipFile(filename)
    for f in z.namelist():
        try:
            os.mkdir(out_dir)
        except FileExistsError:
            pass
        # read inner zip file into bytes buffer
        content = io.BytesIO(z.read(f))
        zip_file = zipfile.ZipFile(content)
        for i in zip_file.namelist():
            zip_file.extract(i, out_dir)


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
    argparser = argparse.ArgumentParser(
        description="Provide model pipeline with necessary arguments: "
                    "- path to training data "
                    "-whatever else comes to my mind later")
    argparser.add_argument('-t', '--train_dir',
                           help='abs or rel path to .txt files with raw training samples.')
    # todo pass path to setup.py and make setup return mount context here
    # todo mount context can depend on _Offline or Run context
    cmd_args, _ = argparser.parse_known_args()
    #
    # train_dir = cmd_args.train_dir
    # if cmd_args.train_dir is None:
    #     # train_dir = DATA_DIR + '03_preprocessed/0.25-0.50/preprocessed'
    #     train_dir = '03_preprocessed/0.25-0.50/preprocessed'

    # # Connect to Workspace and reference Dataset
    # ws = Workspace.from_config()
    # dataset = ws.datasets["preprocessed"]
    #
    # print(dataset.name)
    #
    # # Create mountcontext and mount the dataset
    # mount_ctx = dataset.mount()
    # mount_ctx.start()
    #
    # # Get the mount point
    # train_dir = mount_ctx.mount_point
    # train_dir = "Users/sascha.lange/github.com/hellovertex/Thesis/data/03_preprocessed/0.25-0.50"
    train_dir = "Users/sascha.lange/github.com/hellovertex/Thesis/data/03_preprocessed/0.25-0.50/preprocessed"
    print(f"Train_DIR: {train_dir}")
    print(f'FILES INSIDE: = {os.listdir(train_dir)}')
    # train_dir = DATA_DIR + '03_preprocessed/0.25-0.50/preprocessed'
    # zipfiles = glob.glob(train_dir.__str__() + '/*.zip', recursive=False)
    # print(f"zipfiles = {zipfiles}")
    # [extract(zipfile, out_dir=train_dir) for zipfile in zipfiles]

    dataloaders = get_dataloaders(train_dir)
    train_loader = dataloaders['train']
    test_loader = dataloaders['test']

    mlflow_run = run(train_loader, test_loader)
