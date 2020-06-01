# -------------------------------------------------------------------------
# History:
# [AMS - 200601] created
# -------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def bcnn_loss(c1, c2, f1, y_train, weights):
    """
        Function to calculate weighted 3 term loss function for BCNN
    """

    y_c1_train = l1_labels(y_train)
    y_c2_train = l2_labels(y_train)

    l1 = F.cross_entropy(c1, y_c1_train)
    l2 = F.cross_entropy(c2, y_c2_train)
    l3 = F.cross_entropy(f1, y_train)

    loss = weights[0]*l1 + weights[1]*l2 + weights[2]*l3

    return loss


def l1_labels(labels):

    """
        0: vehicle (0:plane, 1:car, 4:truck)
        1: animal (2:bird, 3:horse)
    """

    l1_labels = np.zeros(labels.size())
    l1_labels[np.where(labels==2)]=1
    l1_labels[np.where(labels==3)]=1

    return torch.tensor(l1_labels, dtype=torch.long)


def l2_labels(labels):

    """
        0: air (0:plane)
        1: ground (1:car, 4:truck)
        2: bird (2:bird)
        3: horse (3:horse)
    """

    l2_labels = np.zeros(labels.size())
    l2_labels[np.where(labels==1)]=1
    l2_labels[np.where(labels==2)]=2
    l2_labels[np.where(labels==3)]=3
    l2_labels[np.where(labels==4)]=1

    return torch.tensor(l2_labels, dtype=torch.long)
