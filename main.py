# -------------------------------------------------------------------------
# History:
# [AMS - 200601] created
# -------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchsummary import summary

import numpy as np

from cifar5 import CIFAR5
from models import BCNN
from utils import bcnn_loss


batch_size    = 4                   # number of samples per mini-batch
imsize        = 50                  # image size
params        = [2,4,5]             # [coarse1, coarse2, fine]
weights       = [0.8,0.1,0.1]       # weights for loss function
lr0           = torch.tensor(1e-3)  # speed of convergence
momentum      = torch.tensor(8e-1)  # momentum for optimizer
decay         = torch.tensor(1e-6)  # weight decay for regularisation
random_seed   = 42

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
print("Device: ",device)

# -----------------------------------------------------------------------------

transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(50),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
            ])


trainset = CIFAR5(root='./cifar5data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = CIFAR5(root='./cifar5data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'horse', 'truck')

# -----------------------------------------------------------------------------

model = BCNN(in_chan=1, params=params, kernel_size=3)
learning_rate = lr0
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)

# -----------------------------------------------------------------------------

summary(model, (1, imsize, imsize))
model = model.to(device)

# -----------------------------------------------------------------------------

epochs = 10
epoch_trainaccs, epoch_testaccs = [], []
epoch_trainloss, epoch_testloss = [], []

for epoch in range(epochs):

    model.train()
    train_losses, train_accs = [], []; acc = 0
    for batch, (x_train, y_train) in enumerate(trainloader):

        model.zero_grad()
        x_train, y_train = x_train.to(device), y_train.to(device)
        c1_pred, c2_pred, f1_pred = model(x_train)

        loss = bcnn_loss(c1_pred, c2_pred, f1_pred, y_train, weights, device=device)
        loss.backward()
        optimizer.step()

        acc = (f1_pred.argmax(dim=-1) == y_train).to(torch.float32).mean()
        train_accs.append(acc.mean().item())
        train_losses.append(loss.item())

    print('Train: {}, Loss: {}, Accuracy: {}'.format(epoch, np.mean(train_losses), np.mean(train_accs)))

    with torch.no_grad():

        model.eval()
        test_losses, test_accs = [], []; acc = 0
        for i, (x_test, y_test) in enumerate(testloader):

            x_test, y_test = x_test.to(device), y_test.to(device)
            c1_testpred, c2_testpred, f1_testpred = model(x_test)

            loss = bcnn_loss(c1_testpred, c2_testpred, f1_testpred, y_test, weights, device=device)

            acc = (f1_testpred.argmax(dim=-1) == y_test).to(torch.float32).mean()
            test_losses.append(loss.item())
            test_accs.append(acc.mean().item())

    print(' Test: {}, Loss: {}, Accuracy: {}'.format(epoch, np.mean(test_losses), np.mean(test_accs)))
    print('---')
    epoch_trainaccs.append(np.mean(train_accs))
    epoch_testaccs.append(np.mean(test_accs))
    epoch_trainloss.append(np.mean(train_losses))
    epoch_testloss.append(np.mean(test_losses))

print("Final test error: ",100.*(1 - epoch_testaccs[-1]))

np.savez("bcnntestloss.npz",np.array(epoch_testloss))
np.savez("bcnntrainloss.npz",np.array(epoch_trainloss))
