# coding=utf-8
"""
PyCharm Editor
Author cyRi-Le
"""
from typing import Tuple

from torchvision.datasets import MNIST, CIFAR10


class FastMNISTLoader(MNIST):
    """
    Copied from a web source
    Implement a faster Loader for MNIST
    This reduce the CPU time during training
    """

    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data.to(device), self.targets.to(device)
        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # Normalize it with the usual MNIST mean and std
        self.data = self.data.sub_(0.1307).div_(0.3081)

    def __getitem__(self, index: int) -> Tuple:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target
