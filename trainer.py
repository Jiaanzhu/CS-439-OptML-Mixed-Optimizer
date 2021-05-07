# coding=utf-8
"""
PyCharm Editor
Author cyRi-Le
"""

import torch
from typing import Callable, List, Iterable, Optional
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
import numpy as np

SEED = 123456789
# seeding pytorch
# From https://pytorch.org/docs/stable/notes/randomness.html
torch.use_deterministic_algorithms(True)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
# # Data loading
BATCH_SIZE = 10

# torchvision.datasets.MNIST outputs a set of PIL images
# We transform them to tensors
transform = transforms.ToTensor()

# Load and transform data
trainset = torchvision.datasets.MNIST('/tmp', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST('/tmp', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# # Data visualization
#
# Let's explore the dataset, especially to determine the dimension of data.

def show_batch(batch):
    im = torchvision.utils.make_grid(batch)
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))


dataiter = iter(trainloader)
images, labels = dataiter.next()

print('Labels: ', labels)
print('Batch shape: ', images.size())
show_batch(images)

# # MLP model

# http://pytorch.org/docs/master/tensors.html#torch.Tensor.view
images.view(BATCH_SIZE, -1).size()


class SequentialMNIST(nn.Module):
    """
    Sequential MNIST
    """

    def __init__(self):
        super(SequentialMNIST, self).__init__()
        self.linear1 = nn.Linear(784, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 10)

    def forward(self, x):
        """
        @override forward method
        :param x:
        :return:
        """
        h_relu = F.relu(self.linear1(x))
        h_relu = F.relu(self.linear2(h_relu))
        y_pred = self.linear3(h_relu)
        return y_pred


def predict(model, images):
    """
    :param model:
    :param images:
    :return:
    """
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)  # TODO: explain why 1
    return predicted


def test(model, testloader):
    correct = 0
    for data in testloader:
        inputs, labels = data
        inputs = inputs.view(-1, 784).to(device)
        labels = labels.to(device)
        pred = predict(model, inputs)
        correct += (pred == labels).sum()
    return 100 * correct / len(testloader)


def train(model, trainloader, valloader, criterion, optimizer, n_epochs=2, interval=1000, device=device):
    val_loss = []
    val_acc = []
    for t in range(n_epochs):
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.view(-1, 784).to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Compute the gradient for each variable
            optimizer.step()  # Update the weights according to the computed gradient

            if not i % interval:
                print(t, i, loss.item())
                acc = test(model, valloader)
                print('Accuracy: %.2f %%' % acc)
    return val_acc, val_loss


model = SequentialMNIST()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
dataiter = iter(testloader)
images, labels = dataiter.next()
show_batch(images)
EPOCHS = 100


def Trainer(batch_size,
            verbose=False):
    val_accuracies, val_losses = [], []
    model = SequentialMNIST()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    # Load and transform data
    trainset = torchvision.datasets.MNIST('/tmp', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST('/tmp', train=False, download=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(
        trainset, [50000, 10000], generator=torch.Generator().manual_seed(SEED))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    val_acc, val_loss = train(model, trainloader, valloader, criterion, optimizer, n_epochs=EPOCHS, interval=1000, device=device)
    val_accuracies.append(val_acc)
    val_losses.append(val_loss)

 trainings = [Trainer(128), Trainer(50)]