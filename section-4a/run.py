import tqdm
import time
import torch
import random
import numpy as np
import torch.nn as nn
from typing import Optional
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST

# define the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# define a fast loader for MNIST
# since we make a lot of test
# this implementation is required
# to avoid CPU bottleneck and
# to train over reasonable time

class FastMNIST(MNIST):
    """
    The dataset is loaded once one the device
    This is done to avoid CPU bottleneck
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data.to(device), self.targets.to(device)
        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)
        # Normalize it with the usual MNIST mean and std
        self.data = self.data.sub_(0.1307).div_(0.3081)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


# we define a seed
SEED = 123456789
# Load and transform data
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class SequentialMNIST(nn.Module):
    """
    Sequential MNIST model
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
        x = x.view(-1, 784)
        h_relu = F.relu(self.linear1(x))
        h_relu = F.relu(self.linear2(h_relu))
        y_pred = self.linear3(h_relu)
        return y_pred


@torch.no_grad()
def accuracy(model,
             loader):
    """
    Compute the accuracy of the model
    :param model: Model to compute accuracy
    :param loader: dataset loader
    :return: accuracy
    """
    correct = 0
    total = 0
    for data in loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        pred = model(inputs)
        correct += (pred.argmax(-1) == labels).sum()
        total += inputs.size(0)
    return 100 * correct / total


@torch.no_grad()
def accuracy_and_loss(model,
                      valloader,
                      criterion):
    """
    Compute the validation loss and accuracy
    :param model:
    :param valloader: Dataset loader
    :param criterion: Loss function used
    :return: Validation loss and accuracy
    """
    correct = 0
    total = 0
    loss = None
    # valloader has in fact 1 array
    # which is the whole validation set
    # this is to be generic
    for data in valloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        preds = model(inputs)
        loss = criterion(preds, labels)
        correct += (preds.argmax(-1) == labels).sum()
        total += inputs.size(0)
    return 100 * correct / total, loss


def train(model,
          traindata,
          traintargets,
          valloader,
          criterion,
          optimizers,
          n_epochs: int,
          interval: Optional[int] = 50,
          keep_last: Optional[bool] = True,
          device=device,
          batches=None):
    """

    :param model: Model to be trained
    :param traindata: Training datapoints
    :param traintargets: Training labels (groundtruth)
    :param valloader: Validation DataLoader
    :param criterion: Loss function
    :param optimizers: List of optimizers
    :param n_epochs: Number of epochs
    :param interval: Number of points to record per epoch
    :param keep_last: Whether to record the last step of the epoch
    :param device: Device on which the model is trained
    :param batches: Array of indices where each element is the batch indices
    :return:
    """
    # put the model on the device
    # all the data are assumed to be on the same device
    model = model.to(device)
    # define empty list for the records
    val_loss = []
    val_acc = []
    epochs = []
    train_loss = []
    train_acc = []
    n_train = len(batches)
    # frequency to record metrics
    period = len(batches) // interval
    for epoch in range(n_epochs):
        for i, data in enumerate(zip(traindata, traintargets)):
            inputs, labels = data
            for opt in optimizers:
                opt.zero_grad()
            outputs = model(inputs)
            # Compute the loss
            loss = criterion(outputs, labels)
            # Compute the gradients
            loss.backward()
            for opt in optimizers:
                opt.step()
            if not i % period or (keep_last and i == n_train - 1):
                # compute and record the validation accuracy and loss
                acc, loss = accuracy_and_loss(model, valloader, criterion)
                val_acc.append(acc.item())
                val_loss.append(loss.item())
                # compute and record the training accuracy and loss
                train_loss.append(loss.item())
                t_acc = accuracy(model, [data]).item()
                train_acc.append(t_acc)
                epochs.append((epoch + 1) + i / n_train)
    return epochs, val_acc, val_loss, train_acc, train_loss


# define the loss function
# and the number of epochs
criterion = nn.CrossEntropyLoss()
# you can with fewer for testing
EPOCHS = 150


# this function is used to compute
# a list of batch indices
# each element of the list (possibly except the last one) has batch size
# elements which are the indices of the batch
def batchify(bsz,
             size):
    """
    Return an array of batch indices
    :param bsz: Batch size
    :param size: Number of points in the dataset
    :return: batches
    """
    indices = np.random.permutation(size)
    new_size = (size // bsz) * bsz
    d_indices = indices[:new_size].reshape((-1, bsz))
    r_indices = indices[new_size:].reshape((1, -1))
    batches = d_indices.tolist() + r_indices.tolist()
    return batches


def Trainer(batch_size,
            optimizers,
            model,
            sample: Optional[int] = 1,
            spacing: Optional[int] = 1):
    """
    Run the tests given a configuration
    We use Adam and SGD
    :param batch_size: Batch size to use
    :param sample: The number of trials to repeat
    :param optimizers: List of optimizers
    :param model: Model to train
    :param spacing: Number of points to record per epoch
    :return: List
    """
    # for repeatability 
    # seeding pytorch and numpy for each configuration 
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    epochs, val_acc, val_loss, train_acc, train_loss = [], [], [], [], []
    # Load validation and training data
    # The test set of MNIST is used as validation set
    valset = FastMNIST('data/MNIST', train=False, download=True)
    trainset = FastMNIST('data/MNIST', train=True, download=True)
    valdata = [valset.data, valset.targets]

    for _ in tqdm.tqdm(range(sample)):
        # compute the batch indices
        batches = batchify(batch_size, 60000)
        # batchify the training data
        # the validation set is given entirely at once
        # to the model since there is no need to batchify it
        traindata = [trainset.data[batch] for batch in batches]
        traintargets = [trainset.targets[batch] for batch in batches]
        # run the training and save the points
        _epochs, _val_acc, _val_loss, _train_acc, _train_loss = train(model,
                                                                      traindata,
                                                                      traintargets,
                                                                      [valdata],
                                                                      criterion,
                                                                      optimizers,
                                                                      n_epochs=EPOCHS,
                                                                      interval=spacing,
                                                                      device=device,
                                                                      batches=batches)
        # add the recorded points
        val_acc.append(_val_acc)
        val_loss.append(_val_loss)
        train_acc.append(_train_acc)
        train_loss.append(_train_loss)
    epochs = _epochs
    return [epochs,
            np.asarray(
                [val_acc,
                 val_loss,
                 train_acc,
                 train_loss]),
            ]


bsz = 256
lr1 = 1e-3
lr2 = 1e-2
# running the first test with Adam alone 
# instantiate a model
model = SequentialMNIST()
# define the optimizers
opt_adam = optim.Adam(model.parameters(), lr=lr1)
optimizers = [opt_adam]
# number of trials
sample = 10
# number of records per epoch
spacing = 50
# tic
start = time.time()
# run the test
x, data_adam = Trainer(model=model, batch_size=bsz, optimizers=optimizers, spacing=spacing, sample=sample)
# toc
duration = time.time() - start
print("[Adam] Mean time per epoch per trial: %.2f s" % (duration / sample / EPOCHS))
x = np.array(x)

# run the second test with SGD alone
# re-instantiate a model
model = SequentialMNIST()
# redefine optimizers
opt_sgd = optim.SGD(model.parameters(), lr=lr2)
optimizers = [opt_sgd]

# tic
start = time.time()
# run the test
_, data_sgd = Trainer(model=model, batch_size=bsz, optimizers=optimizers, spacing=spacing, sample=sample)
# toc
duration = time.time() - start
print("[SGD] Mean time per epoch per trial: %.2f s" % (duration / sample / EPOCHS))

# run the third test with Mix optimizer
# re-instantiate a model
model = SequentialMNIST()
# redefine optimizers
# first layer parameters are 
# trained with adam 
# see report for details 
opt_adam = optim.Adam([model.linear1.weight], lr=lr1)
# one hidden and output layer parameters are trained 
# using SGD 
# see report for details 
opt_sgd = optim.SGD([model.linear2.weight, model.linear3.weight], lr=lr2)
optimizers = [opt_adam, opt_sgd]
# tic
start = time.time()
# run the test
_, data_mix = Trainer(model=model, batch_size=bsz, optimizers=optimizers, spacing=spacing, sample=sample)
# toc
duration = time.time() - start
print("[Mix] Mean time per epoch per trial: %.2f s" % (duration / sample / EPOCHS))

# process the validation data
# for the mixed optimizer
val_mix_mean = data_mix[0].mean(0)
val_mix_max = data_mix[0].max(0)
val_mix_min = data_mix[0].min(0)
# for Adam alone
val_adam_mean = data_adam[0].mean(0)
val_adam_max = data_adam[0].max(0)
val_adam_min = data_adam[0].min(0)
# for SGD alone
val_sgd_mean = data_sgd[0].mean(0)
val_sgd_max = data_sgd[0].max(0)
val_sgd_min = data_sgd[0].min(0)
# foreshortened the plot to show
# points from accuracy greater than 90%
# this is to zoom on convergence
# it is not the same we have used for all configurations
# since values may depend on configuration
threshold = 85
indices = (val_mix_mean > threshold) * (val_sgd_mean > threshold) * (val_adam_mean > threshold)


# since the recorded max and min points
# can be noisy
# there are smoothed with a moving average
# they only serve for uncertainty area
def moving_average(a, n=3):
    """
    We smooth the curves with a moving average
    :param a: List of values
    :param n: Average on n points
    :return: List
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# number of points to average on
# recall that we save at least 50 points per epoch
# and we train over 150 epochs
pts = 100
# process mixed optimizer data
val_mix_min_mov = np.append(moving_average(val_mix_min, pts), val_mix_min[-(pts - 1):])
val_mix_max_mov = np.append(moving_average(val_mix_max, pts), val_mix_max[-(pts - 1):])

# process Adam optimizer data
val_adam_min_mov = np.append(moving_average(val_adam_min, pts), val_adam_min[-(pts - 1):])
val_adam_max_mov = np.append(moving_average(val_adam_max, pts), val_adam_max[-(pts - 1):])

# process SGD optimizer data
val_sgd_min_mov = np.append(moving_average(val_sgd_min, pts), val_sgd_min[-(pts - 1):])
val_sgd_max_mov = np.append(moving_average(val_sgd_max, pts), val_sgd_max[-(pts - 1):])

# define the figure
plt.figure(dpi=600, figsize=(8, 5))
plt.grid()
plt.plot(x[indices], val_sgd_mean[indices], label=f"SGD($l_r={lr2}$)", )
plt.plot(x[indices], val_adam_mean[indices], label=f"Adam($l_r={lr1}$)")
plt.plot(x[indices], val_mix_mean[indices], label="Mix")
plt.xlabel("epoch")
plt.ylabel("validation accuracy [%]")
plt.fill_between(x[indices], val_sgd_min_mov[indices], val_sgd_max_mov[indices], alpha=0.5, antialiased=True)
plt.fill_between(x[indices], val_adam_min_mov[indices], val_adam_max_mov[indices], alpha=0.5, antialiased=True)
plt.fill_between(x[indices], val_mix_min_mov[indices], val_mix_max_mov[indices], alpha=0.5, antialiased=True)
plt.legend()
# please comment this line to
# avoid saving the report
plt.savefig(f"figure-config[Adam={lr1:.2E}]-[SGD={lr2:.2E}].png")
plt.show()
# To try a different configuration
# is suffices to change lr1 and lr2
# Thanks for reading