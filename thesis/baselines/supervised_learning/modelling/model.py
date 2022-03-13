import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1


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


model = Net(input_dim=input_size, output_dim=num_classes, hidden_dim=[512, 512])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train_loader = []  # todo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):


        # possibly reshape

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        # backward
        optimizer.zero_grad()
        loss.backward()
        # gradient descent / adam step
        optimizer.step()


# def check_accuracy(loader, model):
#     num_correct = 0
#     num_samples = 0
#     model.eval()
#     with torch.no_grad():
#         for x,y in loader:
#             x = data.to(device=device)
#             y = targets.to(device=device)
#
#             scores = model(x)
