# -*- coding: utf-8 -*-
import time
import pickle
import torch
import torchmetrics
import timm

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F



class TimmModel(pl.LightningModule):
    """Use networks from the library Timm"""

    def __init__(
        self, network_name, map_size=64, mass_plotter=None, lr=0.001, **kwargs
    ):
        super(TimmModel, self).__init__()
        self.lr = lr
        self.loss = F.mse_loss
        self.mass_plotter = mass_plotter
        self.return_y_in_pred = False
        self.dump_predictions = False

        if network_name[:4] in ("resn", "dens", "ese_", "hrne"):
            self.model = timm.create_model(
                network_name, pretrained=False, num_classes=0
            )
        elif network_name[:4] in ("xcep"):
            self.model = timm.create_model(
                network_name, pretrained=False, num_classes=0, input_size=map_size
            )
        else:
            self.model = timm.create_model(
                network_name, pretrained=False, num_classes=0, img_size=map_size
            )
        x = torch.ones(1, 3, map_size, map_size)
        self.final_shape = self.model(x).shape[-1]
        self.mlp = nn.Sequential(
            nn.Linear(self.final_shape, 64),
            nn.BatchNorm1d(
                64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.Dropout(p=0.20),
            nn.Linear(64, 32),
            nn.BatchNorm1d(
                32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.Dropout(p=0.20),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.mlp(x)
        return x

    def configure_optimizers(self, step_size=1):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)[:, None, None, ...]
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        # return {"loss": loss}
        # Return values to plot graphs
        return {
            "loss": loss,
            "y": y.detach().cpu().numpy(),
            "y_hat": y_hat.detach().cpu().numpy(),
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)[:, None, None, ...]
        val_loss = self.loss(y_hat, y)
        self.log("val_loss", val_loss)
        # return {"val_loss": val_loss}
        return {
            "val_loss": val_loss,
            "y": y.cpu().numpy(),
            "y_hat": y_hat.cpu().numpy(),
        }

    def validation_epoch_end(self, outputs):
        """Plot graphs to writer at the end of each validation epoch"""

        if self.dump_predictions:
            # Dump the predictions for the current epoch
            f = open(
                "validation_epoch_end_{}_{}".format(self.current_epoch, time.time()), "wb"
            )
            pickle.dump(
                [(outputs[i]["y"], outputs[i]["y_hat"]) for i in range(len(outputs))], f
            )
            f.close()

        if self.mass_plotter is not None:
            self.mass_plotter.plot_all(outputs, self.current_epoch, step="validation")

    def training_epoch_end(self, outputs):
        """Plot graphs to writer at the end of each training epoch"""

        if self.dump_predictions:
            # Dump the predictions for the current epoch
            f = open(
                "training_epoch_end_{}_{}".format(self.current_epoch, time.time()), "wb"
            )
            pickle.dump(
                [(outputs[i]["y"], outputs[i]["y_hat"]) for i in range(len(outputs))], f
            )
            f.close()

        if self.mass_plotter is not None:
            self.mass_plotter.plot_all(outputs, self.current_epoch, step="training")

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)[:, None, None, ...]
        if self.return_y_in_pred:
            return y, pred
        return pred

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MResUNet")
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--network_name", type=str, default="resnet50")
        return parent_parser
