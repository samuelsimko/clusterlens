# -*- coding: utf-8 -*-
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import torch
import numpy as np

from models.mresunet import MResUNet
from models.resnet import ResNet

from mapdatamodule import MapDataModule

import matplotlib.pyplot as plt

# from datamodule import GenericDataModule


def vis(args):
    """Visualize model predictions"""

    dm = MapDataModule(
        **vars(args),
        train_dir=args.data_dir,
        val_dir=args.data_dir,
        batch_size=1,
        num_workers=1,
        transform=None
    )
    dm.setup()

    model = MResUNet.load_from_checkpoint(
        lr=1, checkpoint_path=args.checkpoint, map_size=dm.npix
    )
    model.train(False)
    with torch.no_grad():
        for x, y in dm.val_dataloader():
            y_hat = model(x)

            y_hat = (
                y_hat[:, None, :].float().detach().numpy().reshape((dm.npix, dm.npix))
            )
            x = x.detach().numpy().reshape((dm.npix, dm.npix))
            y = y.detach().numpy().reshape((dm.npix, dm.npix))

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
            plt.imshow(np.abs(y_hat - y))
            plt.colorbar()
            plt.title("Absolute Error")
            plt.suptitle(
                "Prediction with MSE = {}".format(
                    np.mean((y_hat.flatten() - y.flatten()) ** 2)
                )
            )

            plt.show()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="A program to train Deep Nets on simulated CMB maps to estimate the masses of galaxy clusters"
    )
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument(
        "--model",
        type=str,
        default="mresunet",
        help="The model to use",
        choices=["mresunet", "resunet"],
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="The checkpoint from which to load the model weights",
        required=True,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="The path to the directory containing the prediction data",
        required=True,
    )

    temp_args, _ = parser.parse_known_args()

    if temp_args.model == "mresunet":
        print("adding")
        parser = MResUNet.add_model_specific_args(parser)

    args = parser.parse_args()

    vis(args)
