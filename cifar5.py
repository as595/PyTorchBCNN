# -------------------------------------------------------------------------
# History:
# [AMS - 200601] created
# -------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import numpy as np


class CIFAR5(datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super(CIFAR5, self).__init__(*args, **kwargs)

        #exclude_list = ['cat', 'deer', 'dog', 'frog', 'ship']
        exclude_list = [3, 4, 5, 6, 8]

        if exclude_list == []:
            return

        if self.train:
            labels = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

            labels = self.renumber_labels(labels[mask])

            self.data = self.data[mask]
            self.targets = labels.tolist()

        else:
            labels = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

            labels = self.renumber_labels(labels[mask])

            self.data = self.data[mask]
            self.targets = labels.tolist()


    def renumber_labels(self,labels):

        """
            Input:
            0: plane, 1: car, 2: bird, 7: horse, 9: truck
            Output:
            0: plane, 1: car, 2: bird, 3: horse, 4: truck
        """

        labels[np.where(labels==7)]=3
        labels[np.where(labels==9)]=4

        return labels
