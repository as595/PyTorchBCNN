# -------------------------------------------------------------------------
# History:
# [AMS - 200601] created
# [AMS - 200601] added lenet for comparison
# [AMS - 200601] added BaseCNN for comparison
# -------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from layers import ConvBlock, CoarseBlock


class BCNN(nn.Module):
    def __init__(self, in_chan, params, kernel_size=3):
        super(BCNN, self).__init__()

        c1_targets, c2_targets, out_chan = params

        self.convblock1 = ConvBlock(in_channels=in_chan, hidden=32, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, hidden=128, out_channels=128)
        self.coarse1    = CoarseBlock(in_features=128*12*12, hidden=128, out_features=c1_targets)
        self.convblock3 = ConvBlock(in_channels=128, hidden=256, out_channels=256)
        self.coarse2    = CoarseBlock(in_features=256*6*6, hidden=1024, out_features=c2_targets)
        self.convblock4 = ConvBlock(in_channels=256, hidden=512, out_channels=512)
        self.coarse3    = CoarseBlock(in_features=512*3*3, hidden=1024, out_features=out_chan)


    def forward(self, x):

        x = self.convblock1(x)
        x = self.convblock2(x)

        l1 = x.view(x.size()[0], -1)
        c1, c1_pred = self.coarse1(l1)

        x = self.convblock3(x)

        l2 = x.view(x.size()[0], -1)
        c2, c2_pred = self.coarse2(l2)

        x = self.convblock4(x)

        l3 = x.view(x.size()[0], -1)
        f1, f1_pred = self.coarse3(l3)

        return c1, c2, f1

# -----------------------------------------------------------------------------

class BaseCNN(nn.Module):
    def __init__(self, in_chan, params, kernel_size=3):
        super(BaseCNN, self).__init__()

        c1_targets, c2_targets, out_chan = params

        self.convblock1 = ConvBlock(in_channels=in_chan, hidden=32, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, hidden=128, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, hidden=256, out_channels=256)
        self.convblock4 = ConvBlock(in_channels=256, hidden=512, out_channels=512)
        self.coarse3    = CoarseBlock(in_features=512*3*3, hidden=1024, out_features=out_chan)

    def forward(self, x):

        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = x.view(x.size()[0], -1)
        f1, f1_pred = self.coarse3(x)

        return f1, f1_pred


# -----------------------------------------------------------------------------


class LeNet(nn.Module):
    def __init__(self, in_chan, out_chan, imsize, kernel_size=5):
        super(LeNet, self).__init__()

        z = 0.5*(imsize - 2)
        z = int(0.5*(z - 2))

        self.conv1 = nn.Conv2d(in_chan, 6, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size, padding=1)
        self.fc1   = nn.Linear(16*z*z, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, out_chan)
        self.drop  = nn.Dropout(p=0.5)

        self.init_weights()

    def init_weights(self):
        # weight initialisation:
        # following: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        for m in self.modules():
            if isinstance(m, nn.Linear):
                y = m.in_features
                nn.init.uniform_(m.weight, -np.sqrt(3./y), np.sqrt(3./y))
                nn.init.constant_(m.bias, 0)

    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)

        return x
