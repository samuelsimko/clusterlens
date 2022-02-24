# -*- coding: utf-8 -*-
import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from mresunet import MResUNet
from lensingmapdataset import LensingMapDataset
from lensingmapdataset import initialize_camb


# Initialize camb
cambinifile = "planck_2018_acc"
results = initialize_camb(cambinifile)

# Number of generated maps
nsims = 100

# M200 masses of generated maps
m200_list = [1e14, 2e14, 3e14, 4e14]

# Number of pixels
npix = 32

# Physical size of a pixel in arcmin
lpix_amin = 0.3

# Maximum multipole used to generate the CMB maps from the CMB power spectra
ellmaxsky = 6000

# Redshift
z = 1

# Profile name
profname = "nfw"

# Training parameters
learning_rate = 3e-2
batch_size = 20
epochs = 20
loss_fn = nn.MSELoss()

transform = None

# Generate dataset
training_data = LensingMapDataset(
    cambinifile,
    results,
    npix,
    nsims,
    lpix_amin,
    profname,
    ellmaxsky,
    m200_list,
    z,
    transform=transform,
)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define network
net = MResUNet(npix, device).to(device).double()

# Define optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    """Training loop"""
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X[:, None, :]
        y = y[:, None, :]
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 2 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# Train for multiple epochs
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, net, loss_fn, optimizer)
print("Done!")
