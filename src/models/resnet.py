# -*- coding: utf-8 -*-
from torch import nn
import torch.nn.functional as F


# Both the ResNet and ResBlock codes were taken and adapted from Francois Fleuret's UNIGE/EPFL
# "Deep Learning" course Residual Networks implementations. (https://fleuret.org/dlc)


class ResBlock(nn.Module):
    """A residual block. Performs batch normalization, convolution and residual connection."""

    def __init__(self, nb_channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(
            nb_channels, nb_channels, kernel_size, padding=(kernel_size - 1) // 2
        )
        self.bn1 = nn.BatchNorm2d(nb_channels)
        self.conv2 = nn.Conv2d(
            nb_channels, nb_channels, kernel_size, padding=(kernel_size - 1) // 2
        )
        self.bn2 = nn.BatchNorm2d(nb_channels)

    def forward(self, x):
        y = self.bn1(self.conv1(x))
        y = F.relu(y)
        y = self.bn2(self.conv2(y))
        y += x
        y = F.relu(y)
        return y


class ResNet(nn.Module):
    """A Residual Network

    Args:
        nb_channels: the number of channels in each ResBlock
        kernel_size: the size of the kernel used in convolutions
        nb_blocks: the number of residual blocks used
        npix: the width/height of the square matrix input.
    """

    def __init__(self, nb_channels, kernel_size, nb_blocks, npix):
        super().__init__()
        self.conv0 = nn.Conv2d(1, nb_channels, kernel_size=1)
        self.resblocks = nn.Sequential(
            *(ResBlock(nb_channels, kernel_size) for _ in range(nb_blocks))
        )
        self.conv1 = nn.Conv2d(
            in_channels=nb_channels, out_channels=1, kernel_size=3, padding=1
        )
        self.lin = nn.Linear(npix, npix)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = self.resblocks(x)
        x = self.conv1(x)
        x = self.lin(x)
        return x
