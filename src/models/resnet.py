# -*- coding: utf-8 -*-
import pytorch_lightning as pl
import torch
import torch.nn as nn
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


class ResNet(pl.LightningModule):
    """A Residual Network

    Args:
        nb_channels: the number of channels in each ResBlock
        kernel_size: the size of the kernel used in convolutions
        nb_blocks: the number of residual blocks used
        npix: the width/height of the square matrix input.
    """

    def __init__(
        self,
        input_channels,
        nb_channels,
        kernel_size,
        nb_blocks,
        npix,
        loss="mse",
        output_type="kappa_map",
        lr=0.001,
        final_channels=1,
        mass_plotter=None,
        **kwargs
    ):
        super().__init__()
        self.lr = lr
        self.mass_plotter = mass_plotter
        self.output_type = output_type

        if loss == "msle":
            self.loss = lambda x, y: F.mse_loss(
                torch.log(x + 1e-14), torch.log(y + 1e-14)
            )
        else:
            self.loss = F.mse_loss

        self.conv0 = nn.Conv2d(input_channels, nb_channels, kernel_size=1)
        self.resblocks = nn.Sequential(
            *(ResBlock(nb_channels, kernel_size) for _ in range(nb_blocks))
        )
        self.conv1 = nn.Conv2d(
            in_channels=nb_channels,
            out_channels=final_channels,
            kernel_size=3,
            padding=1,
        )

        if ["mass"] == self.output_type:
            self.lin = nn.Linear(in_features=npix * npix, out_features=1)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = self.resblocks(x)
        x = self.conv1(x)
        if ["mass"] == self.output_type:
            x = self.lin(x.view(x.shape[0], -1)).view((x.shape[0], 1, 1, 1))
        return x

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ResNet")
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--nb_channels", type=int, default=16)
        parser.add_argument("--kernel_size", type=int, default=3)
        parser.add_argument("--nb_blocks", type=int, default=25)
        return parent_parser

    def configure_optimizers(self, step_size=1):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
        return [optimizer]  # , [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        if "mass" in self.output_type:
            # Return values to plot graphs
            return {
                "loss": loss,
                "y": y.detach().cpu().numpy(),
                "y_hat": y_hat.detach().cpu().numpy(),
            }
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss(y_hat, y)
        self.log("val_loss", val_loss)

        if "mass" in self.output_type:
            # Return values to plot graphs
            return {
                "val_loss": val_loss,
                "y": y.cpu().numpy(),
                "y_hat": y_hat.cpu().numpy(),
            }
        return {"val_loss": val_loss}

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        return pred

    def validation_epoch_end(self, outputs):
        """Plot graphs to writer at the end of each validation epoch"""

        if "mass" in self.output_type and self.mass_plotter is not None:
            self.mass_plotter.plot_all(outputs, self.current_epoch, step="validation")
            return

    def training_epoch_end(self, outputs):
        """Plot graphs to writer at the end of each training epoch"""

        if "mass" in self.output_type and self.mass_plotter is not None:
            self.mass_plotter.plot_all(outputs, self.current_epoch, step="training")
            return
