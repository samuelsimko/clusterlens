# -*- coding: utf-8 -*-
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


from models.mresunet import MResUNet
from models.resnet import ResNet

from mapdatamodule import MapDataModule


checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="checkpoints",
    filename="clusterlens-{epoch:02d}-{val_loss:.2f}",
    mode="min",
    # save_top_k=3,
    save_last=True,
)

logger = TensorBoardLogger(name="my_model", save_dir="logs")


def main(args):

    dm = MapDataModule(**vars(args), transform=None)
    dm.setup()

    model = MResUNet(**vars(args), map_size=dm.npix)
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
        "--model",
        type=str,
        default="mresunet",
        help="The model to use",
        choices=["mresunet", "resunet"],
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        help="The path to the directory containing the training data",
        required=True,
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        help="The path to the directory containing the validation data",
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

    temp_args, _ = parser.parse_known_args()

    if temp_args.model == "mresunet":
        print("adding")
        parser = MResUNet.add_model_specific_args(parser)

    args = parser.parse_args()

    main(args)
