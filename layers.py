import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):

    """
        Neural network block containing two convolutional layers
        each with a ReLU activation function and batch normalisation, followed
        by a max pooling layer.
    """

    def __init__(self, in_channels, hidden=32, out_channels=64, kernel_size=3):
        """
            :param in_channels: Number of input features
            :param hidden: Dimension of hidden layer
            :param out_channels: Desired number of output features
            :param kernel_size:
        """

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, kernel_size=kernel_size, out_channels=hidden, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(num_features=hidden)
        self.conv2 = nn.Conv2d(in_channels=hidden, kernel_size=kernel_size, out_channels=out_channels, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(num_features=out_channels)
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool2d(2)

    def forward(self, x):
        """
            :param x: Input data of shape
            :return: Output data of shape
        """

        x = self.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.relu(self.conv2(x))
        x = self.bn2(x)

        return self.pool(x)



class CoarseBlock(nn.Module):
    """
        Neural network block for classification using three fully connected
        layers: two with ReLU activation and batch normalisation followed by a
        dropout before the output layer.
    """

    def __init__(self, in_features, hidden, out_features):
        """
            :param in_channels: Number of input features
            :param hidden: Dimension of hidden layer
            :param out_channels: Desired number of output features

        """

        super(CoarseBlock, self).__init__()

        self.fc1  = nn.Linear(in_features=in_features, out_features=hidden)
        self.fc2  = nn.Linear(in_features=hidden, out_features=hidden)
        self.fc3  = nn.Linear(in_features=hidden, out_features=out_features)
        self.bn1  = nn.BatchNorm1d(num_features=hidden)
        self.bn2  = nn.BatchNorm1d(num_features=hidden)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.drop(x)
        x = self.fc3(x)

        return x, F.softmax(x,dim=1)
