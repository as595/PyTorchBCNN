# PyTorchBCNN
PyTorch implementation of hierarchical classification for CIFAR5. 


The [CIFAR5 dataset](./cifar5.py) is a subset of the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and contains the target classes: **plane, car, bird, horse & truck**. 

The hierarchical classification scheme is shown in the following diagram:

![](/media/CIFAR5.png)

To run:

```python
python main.py
```

The code uses the [torch.transforms]() library to convert the CIFAR5 input images from dimensions of (3,32,32) to (1,50,50).

User defined parameters are set at the start of the [main.py](./main.py) script:

```python
batch_size    = 4                   # number of samples per mini-batch
imsize        = 50                  # image size
params        = [2,4,5]             # [coarse1, coarse2, fine]
weights       = [0.8,0.1,0.1]       # weights for loss function
lr0           = torch.tensor(1e-3)  # speed of convergence
momentum      = torch.tensor(8e-1)  # momentum for optimizer
decay         = torch.tensor(1e-6)  # weight decay for regularisation
random_seed   = 42
```

This runs the BCNN defined in [models.py](./models.py), which has the structure:

![](/media/BCNN.png)
