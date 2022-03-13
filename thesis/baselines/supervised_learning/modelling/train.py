import mlflow
import mlflow.pytorch
from mlflow.utils.environment import _mlflow_conda_env
import warnings
import cloudpickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        current_dim = input_dim
        self.layers = nn.ModuleList()
        for hdim in hidden_dim:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        out = F.softmax(self.layers[-1](x))
        return out


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            # Use MLflow logging
            mlflow.log_metric("epoch_loss", loss.item())


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print("\n")
    print("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    # Use MLflow logging
    mlflow.log_metric("average_loss", test_loss)


class Args(object):
    pass


# Training settings
args = Args()
setattr(args, 'batch_size', 64)
setattr(args, 'test_batch_size', 1000)
setattr(args, 'epochs', 3)
setattr(args, 'lr', 0.01)
setattr(args, 'momentum', 0.5)
setattr(args, 'no_cuda', True)
setattr(args, 'seed', 1)
setattr(args, 'log_interval', 10)
setattr(args, 'save_model', True)

use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
# Use Azure Open Datasets for MNIST dataset


def driver(train_loader, test_loader):
    # warnings.filterwarnings("ignore")
    # # Dependencies for deploying the model
    # pytorch_index = "https://download.pytorch.org/whl/"
    # pytorch_version = "cpu/torch-1.1.0-cp36-cp36m-linux_x86_64.whl"
    # deps = [
    #     "cloudpickle=={}".format(cloudpickle.__version__),
    #     pytorch_index + pytorch_version,
    # ]
    with mlflow.start_run() as run:
        model = Net().to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum)
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(args, model, device, test_loader)
        # Log model to run history using MLflow
        if args.save_model:
            model_env = _mlflow_conda_env(additional_pip_deps=deps)
            mlflow.pytorch.log_model(model, "model", conda_env=model_env)
    return run