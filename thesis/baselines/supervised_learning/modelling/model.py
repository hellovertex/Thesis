import os
import pickle

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def train(args, model, device, train_loader, optimizer, epoch, writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                )
            )
            step = epoch * len(train_loader) + batch_idx
            log_scalar(writer, "train_loss", loss.data.item(), step, args.output_dir)
        if (batch_idx % args.log_artifacts_interval == 0) and args.save_model:
            print(f'Logging model artifacts to {args.output_dir}...')
            log_artifacts(model, args.output_dir)


def test(epoch, args, model, test_loader, train_loader, writer):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(
                output, target, reduction="sum"
            ).data.item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), test_accuracy
        )
    )
    step = (epoch + 1) * len(train_loader)
    log_scalar(writer, "test_loss", test_loss, step, args.output_dir)
    log_scalar(writer, "test_accuracy", test_accuracy, step, args.output_dir)


def log_scalar(writer, name, value, step, output_dir):
    """Log a scalar value to both MLflow and TensorBoard"""
    writer.add_scalar(name, value, step)
    mlflow.log_metric(name, value)
    # Upload the TensorBoard event logs as a run artifact
    print("Uploading TensorBoard events as a run artifact...")
    mlflow.log_artifacts(output_dir, artifact_path=output_dir)


def log_artifacts(model, output_dir):
    # Log the model as an artifact of the MLflow run.
    print("\nLogging the trained model as a run artifact...")
    mlflow.pytorch.log_model(model, artifact_path=output_dir, pickle_module=pickle)
    print(
        "\nThe model is logged at:\n%s" % os.path.join(mlflow.get_artifact_uri(), "pytorch-model")
    )
