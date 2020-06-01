import torch
import torch.nn as nn
import torch.nn.functional as F
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
