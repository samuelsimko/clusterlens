# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

import pytorch_lightning as pl
import torch

from typing import Sequence
from dataclasses import dataclass


@dataclass
class MassPlotter:
    """Helper class to plot statistics after every validation_epoch_end"""

    logger: pl.loggers.TensorBoardLogger
    mass_original_mean: Sequence[float]
    mass_original_std: Sequence[float]

    def destandardize(self, mass_list):
        values = (
            np.exp(mass_list * self.mass_original_std + self.mass_original_mean) * 500
        )
        return values

    def plot_all(self, outputs, current_epoch, destandardize=True):

        y_hat = torch.Tensor(np.concatenate([tmp["y_hat"] for tmp in outputs]))
        y = torch.Tensor(np.concatenate([tmp["y"] for tmp in outputs]))

        if destandardize:
            y, y_hat = self.destandardize(y), self.destandardize(y_hat)

        # Show table of statistics of prediction
        pred_std_mean = []
        print(y_hat.cpu().numpy())
        for i in sorted(np.unique(y)):
            tmp = y_hat[(y == i)]
            pred_std_mean.append(torch.std_mean(tmp))
        pred_std_mean = np.array(pred_std_mean)

        fig, axs = plt.subplots(1, 1)
        axs.axis("off")
        axs.table(
            cellText=np.array(pred_std_mean),
            rowLabels=sorted(np.unique(y)),
            colLabels=["std", "mean"],
            loc="center",
        )
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        self.logger.experiment.add_figure("Prediction table", fig, current_epoch)

        fig, axs = plt.subplots(1, 1)
        axs.plot(
            y.cpu().numpy().flatten(),
            y.cpu().numpy().flatten(),
            "xr",
            label="Ideal predictions",
        )
        axs.errorbar(
            x=sorted(np.unique(y)),
            y=np.nan_to_num(pred_std_mean[:, 1], nan=0),
            yerr=np.nan_to_num(pred_std_mean[:, 0], nan=0),
            linestyle="None",
            marker="^",
            label="Predictions",
        )
        axs.legend()
        axs.set_yscale("log")
        axs.set_yscale("log")
        plt.xlabel("Real mass")
        plt.ylabel("Predicted mass")
        plt.title("Predictions for validation set, epoch {}".format(current_epoch))

        self.logger.experiment.add_figure("Prediction plot", fig, current_epoch)
