# -*- coding: utf-8 -*-
import os
import sys

import torch

import matplotlib.pyplot as plt
import numpy as np

from os import path as op
from torch import nn
from torch.utils.data import DataLoader


from models.mresunet import MResUNet
from models.resnet import ResNet
from data_parameters import small_data_params
from directories import TRAINING_LENSIT_DIR, VALIDATION_LENSIT_DIR, CAMB_INI_DIR

os.environ["LENSIT"] = VALIDATION_LENSIT_DIR

from lensingmapdataset import LensingMapDataset
from lensingmapdataset import initialize_camb

if len(sys.argv) < 2:
    print("Usage: python {} model_name".format(sys.argv[0]))
    exit(1)

# Initialize camb
os.environ["CAMBINIDIR"] = CAMB_INI_DIR
cambinifile = "planck_2018_acc"
results = initialize_camb(cambinifile)

# Training parameters
batch_size = 1
loss_fn = nn.MSELoss()

data_params = small_data_params
transform = None

# Generate dataset
data = LensingMapDataset(
    cambinifile,
    results,
    **data_params,
    transform=transform,
)

dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

os.environ["LENSIT"] = VALIDATION_LENSIT_DIR

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define network
npix = data_params["npix"]
# model = ResNet(16, 3, 25, data_params["npix"]).to(device).float()
model = MResUNet(npix, device).to(device).float()
model.load_state_dict(
    torch.load(op.join(os.getcwd(), sys.argv[1]), map_location=device)
)
model.eval()

model.train(False)

for x, y in dataloader:
    x = x[:, None, :].float()
    y = y[:, None, :].float()

    y_hat = model(x).detach().numpy().reshape((npix, npix))
    x = x.detach().numpy().reshape((npix, npix))
    y = y.detach().numpy().reshape((npix, npix))

    plt.subplot(1, 4, 1)
    plt.imshow(x)
    plt.title("Lensed map")
    plt.colorbar()
    plt.subplot(1, 4, 2)
    plt.imshow(y)
    plt.title("Kappa map")
    plt.colorbar()
    plt.subplot(1, 4, 3)
    plt.imshow(y_hat)
    plt.colorbar()
    plt.title("MResUNet prediction")
    plt.subplot(1, 4, 4)
    plt.imshow(np.abs(y_hat - y) / y)
    plt.colorbar()
    plt.title("Relative Error")

    plt.show()
