# coding=utf-8
"""
PyCharm Editor
Author cyRi-Le
"""
import torch
from typing import Optional
import torch.nn.functional as F


class CNN(torch.nn.Module):
    def __init__(self, drop_out: Optional[float] = None):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = torch.nn.Linear(3 * 3 * 64, 256)
        self.fc2 = torch.nn.Linear(256, 10)
        self.drop_out = drop_out if drop_out is not None else 0.5

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 3 * 3 * 64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class FFN(torch.nn.Module):

    def __init__(self):
        super(FFN, self).__init__()
        self.linear1 = torch.nn.Linear(784, 128)
        self.linear2 = torch.nn.Linear(128, 64)
        self.linear3 = torch.nn.Linear(64, 10)

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