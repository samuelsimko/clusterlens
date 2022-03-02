# -*- coding: utf-8 -*-
import os
from datetime import datetime

import torch
import torchvision

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.mresunet import MResUNet
from models.resnet import ResNet
from data_parameters import small_data_params
from directories import TRAINING_LENSIT_DIR, VALIDATION_LENSIT_DIR, CAMB_INI_DIR

os.environ["LENSIT"] = TRAINING_LENSIT_DIR

from lensingmapdataset import LensingMapDataset
from lensingmapdataset import initialize_camb

writer = SummaryWriter()

# Initialize camb
os.environ["CAMBINIDIR"] = CAMB_INI_DIR
cambinifile = "planck_2018_acc"
results = initialize_camb(cambinifile)

# Training parameters
learning_rate = 1e-3
batch_size = 16
epochs = 3
loss_fn = nn.MSELoss()

data_params = small_data_params
transform = None

# Generate dataset
training_data = LensingMapDataset(
    cambinifile,
    results,
    **data_params,
    transform=transform,
)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

os.environ["LENSIT"] = VALIDATION_LENSIT_DIR

# Generate validation dataset
validation_data = LensingMapDataset(
    cambinifile,
    results,
    **data_params,
    transform=transform,
)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define network
# net = ResNet(16, 3, 25, data_params["npix"]).to(device).float()
model = MResUNet(data_params["npix"], device).to(device).float()
model_name = "MResUNet"

images, labels = next(iter(train_dataloader))

grid = torchvision.utils.make_grid(images)
writer.add_image("images", grid[0, :, :], 0, dataformats="HW")
writer.add_graph(model, images[:, None, :].float())
writer.close()

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    """Training loop"""
    size = len(dataloader.dataset)
    running_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X[:, None, :].float()
        y = y[:, None, :].float()
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    avg_loss = running_loss / len(training_data)
    print("Training loss average: ", avg_loss)
    return avg_loss


def validation_loop(dataloader, model, loss_fn):
    """Training loop"""
    running_vloss = 0.0
    for _, (X, y) in enumerate(dataloader):
        X = X[:, None, :].float()
        y = y[:, None, :].float()
        pred = model(X)
        vloss = loss_fn(pred, y)
        running_vloss += vloss.item()

    avg_vloss = running_vloss / len(validation_data)
    print("Validation loss average: ", avg_vloss)
    return avg_vloss


# Train for multiple epochs
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    model.train(True)
    avg_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
    writer.add_scalar("loss/train", avg_loss, t)
    model.train(False)
    avg_vloss = validation_loop(validation_dataloader, model, loss_fn)
    writer.add_scalar("loss/validation", avg_vloss, t)

# Save model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = "{}_{}".format(model_name, timestamp)
torch.save(model.state_dict(), model_path)

print("Done!")
