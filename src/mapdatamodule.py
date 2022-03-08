# -*- coding: utf-8 -*-
import os.path as op
import numpy as np
import pickle

import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MapDataset(Dataset):
    """Map Dataset"""

    def __init__(
        self, path, transform=None, output_type="kappa_map", input_type="tmap"
    ):
        super().__init__()
        self.args = pickle.load(open(op.join(path, "args"), "rb"))
        self.npix = self.args["npix"]
        self.maps = np.load(op.join(path, "maps.npy"), allow_pickle=True)
        self.kappa_maps = np.load(op.join(path, "kappa_maps.npy"), allow_pickle=True)
        self.transform = transform
        self.output_type = output_type
        self.input_type = input_type

    def __len__(self):
        return len(self.args["mass"]) * self.args["nsims"]

    def __getitem__(self, idx):

        sample = []
        if self.input_type == "t_map":
            sample.append(torch.from_numpy(self.maps[idx][0][0]).float()[None, :])
        elif self.input_type == "teb_maps":
            sample.append(torch.from_numpy(self.maps[idx][:-1][0]).float())

        if self.output_type == "kappa_map":
            sample.append(
                torch.from_numpy(self.kappa_maps[self.maps[idx][-1]][0]).float()[
                    None, :
                ]
            )
        elif self.output_type == "mass":
            sample.append(
                torch.Tensor([self.kappa_maps[self.maps[idx][-1]][-1]]).float()[None, :]
            )

        if self.transform:
            sample = self.transform(sample)

        return sample


class MapDataModule(pl.LightningDataModule):
    """Lensing Maps Data Module"""

    def __init__(
        self,
        train_dir,
        val_dir,
        batch_size,
        transform,
        num_workers,
        output_type,
        input_type,
        **args
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers
        self.output_type = output_type
        self.input_type = input_type
        self.npix = pickle.load(open(op.join(train_dir, "args"), "rb"))["npix"]

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = MapDataset(
                self.train_dir,
                self.transform,
                self.output_type,
                input_type=self.input_type,
            )
        if stage == "val" or stage is None:
            self.val_dataset = MapDataset(
                self.val_dir,
                self.transform,
                self.output_type,
                input_type=self.input_type,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
