# -*- coding: utf-8 -*-
import random
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.transforms import transforms

from mapdatamodule import MapDataModule
from models.mresunet import MResUNet
from models.resnet import ResNet


def get_std_mean():
    """Return the std and the mean of the training dataset features"""
    dm = MapDataModule(**vars(args), transform=None)
    dm.setup()
    dm.batch_size = len(dm.train_dataset)
    dl = dm.train_dataloader()
    x, _ = next(iter(dl))
    return torch.std_mean(x, [0, 2, 3])


def main(args):

    if not args.seed:
        args.seed = random.randint(0, 2**32 - 1)
    print("Seeding everything with seed {}...".format(args.seed))
    seed_everything(args.seed)

    checkpoint_name = "_".join(
        [args.model, args.loss, args.input_type, args.output_type, *args.train_dirs]
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename=checkpoint_name + "-{epoch:02d}-{val_loss:.4f}",
        mode="min",
        # save_top_k=3,
        # save_last=True,
    )

    logger = TensorBoardLogger(name=checkpoint_name, save_dir="logs")

    if not args.std_mean:
        # Get mean and std of training datasets
        x_std, x_mean = get_std_mean()
        print("std: {}, mean: {}".format(x_std, x_mean))
    else:
        x_std, x_mean = args.std_mean
        print("std: {}, mean: {}".format(x_std, x_mean))

    transform = transforms.Compose(
        [
            transforms.Normalize(mean=x_mean, std=x_std),
            transforms.Lambda(
                lambda x: (torch.transpose(x, -2, -1) if torch.rand(1) < 0.5 else x)
            ),
        ]
    )

    dm = MapDataModule(**vars(args), transform=transform)
    dm.setup()

    if args.model == "mresunet":
        model = MResUNet(
            **vars(args),
            map_size=dm.npix,
            input_channels=(3 if dm.input_type == "tqu_maps" else 1)
        )
    else:
        model = ResNet(
            **vars(args),
            npix=dm.npix,
            input_channels=(3 if dm.input_type == "tqu_maps" else 1)
        )

    trainer = Trainer.from_argparse_args(
        args, callbacks=[checkpoint_callback], logger=logger
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="A program to train Deep Nets on simulated CMB maps to estimate the masses of galaxy clusters"
    )
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="The seed used for the training (random by default)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mresunet",
        help="The model to use",
        choices=["mresunet", "resnet"],
    )
    parser.add_argument(
        "--train_dirs",
        nargs="+",
        type=str,
        help="The paths to the directories containing the training data",
        required=True,
    )
    parser.add_argument(
        "--val_dirs",
        nargs="+",
        type=str,
        help="The paths to the directories containing the validation data",
        required=True,
    )
    parser.add_argument(
        "--batch_size", type=int, help="The size of a batch", default=16
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="The number of workers in the dataloader",
        default=1,
    )
    parser.add_argument(
        "--output_type",
        default="kappa_map",
        help="The output of the neural net",
        choices=["kappa_map", "mass", "both"],
    )
    parser.add_argument(
        "--input_type",
        default="tmap",
        help="The input of the neural net",
        choices=["t_map", "tqu_maps"],
    )
    parser.add_argument(
        "--std_mean",
        nargs=2,
        type=float,
        help="The mean and the standard deviation to be used for the input normalization",
        default=None,
    )
    parser.add_argument(
        "--rotate_images",
        help="Rotate the input images by 90 degrees with a probability of 0.5",
        action="store_true",
    )
    parser.add_argument(
        "--loss",
        help="The loss function to be used for the neural net",
        choices=["mse", "msle"],
        default="mse",
    )

    temp_args, _ = parser.parse_known_args()

    if temp_args.model == "mresunet":
        parser = MResUNet.add_model_specific_args(parser)

    if temp_args.model == "resnet":
        parser = ResNet.add_model_specific_args(parser)

    args = parser.parse_args()

    main(args)
