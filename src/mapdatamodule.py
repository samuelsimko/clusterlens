# -*- coding: utf-8 -*-
import os
import pickle
import random

import numpy as np
import pytorch_lightning as pl
import torch
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader, Dataset


class MapDataset(Dataset):
    """Map Dataset"""

    def __init__(
        self,
        paths,
        transform=None,
        output_type="kappa_map",
        input_type="tmap",
        crop=None,
        replace_qu=None,
        masses_mean=None,
        masses_std=None,
        keep_mass_smaller_than=None,
        ragh=False,
    ):
        super().__init__()

        self.transform = transform
        self.output_type = output_type
        self.input_type = input_type
        self.crop = crop
        self.replace_qu = replace_qu
        self.keep_mass_smaller_than = keep_mass_smaller_than
        self.print_first_time = True
        self.ragh = False

        self.args = []
        self.maps = []
        self.kappa_maps = []
        self.unl_maps = []
        self.len_maps = []
        self.masses = []

        if self.ragh:
            print("RAGH")
            for path in paths:
                self.args.append(pickle.load(open(os.path.join(path, "args"), "rb")))
                self.maps.append(
                    np.load(os.path.join(path, "maps.npy"), allow_pickle=True)
                )
                self.masses.append(
                    np.load(os.path.join(path, "masses.npy"), allow_pickle=True)
                )
                self.npix = self.args[0]["npix"]
        else:
            for path in paths:
                self.args.append(pickle.load(open(os.path.join(path, "args"), "rb")))
                self.maps.append(
                    np.load(os.path.join(path, "maps.npy"), allow_pickle=True)
                )
                self.kappa_maps.append(
                    np.load(os.path.join(path, "kappa_maps.npy"), allow_pickle=True)
                )
                # self.maps.append(np.load(os.path.join(path, "teb_maps.npy"), allow_pickle=True))
                # self.len_maps.append(
                # np.load(os.path.join(path, "len_maps.npy"), allow_pickle=True)
                # )
                # self.unl_maps.append(
                # np.load(os.path.join(path, "unl_maps.npy"), allow_pickle=True)
                # )
                # self.len_maps.append(
                # np.load(os.path.join(path, "len_maps.npy"), allow_pickle=True)
                # )

                if self.keep_mass_smaller_than is not None:
                    drop_idx = []
                    for idx in range(len(self.maps[-1])):
                        if (
                            self.kappa_maps[-1][self.maps[-1][idx, -1], -1]
                            < self.keep_mass_smaller_than
                        ):
                            drop_idx.append(idx)
                    drop_idx = np.array(drop_idx)
                    self.maps[-1] = self.maps[-1][drop_idx]
                    # self.unl_maps[-1] = self.unl_maps[-1][drop_idx]
                    # self.len_maps[-1] = self.len_maps[-1][drop_idx]

                self.npix = self.args[0]["npix"]

        # Get cumulated sum of the number of maps in each path - 1
        self.len_cumsum = np.cumsum([len(x) for x in self.maps]) - 1

        self.len = self.len_cumsum[-1] + 1

        self.masses_mean = masses_mean
        self.masses_std = masses_std

        self.mass_normalize = Normalize(self.masses_mean, self.masses_std)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        # Get index to use within self.maps
        map_idx = np.searchsorted(self.len_cumsum, idx)
        if map_idx != 0:
            idx -= self.len_cumsum[map_idx - 1] + 1

        sample = []

        if self.ragh:
            # Maps from Raghanuthan's library
            sample.append(torch.from_numpy(self.maps[map_idx][idx]).float())
            sample.append(
                (
                    torch.from_numpy(np.array([self.masses[map_idx][idx]])).float()[
                        None, None, :
                    ]
                    - 12.5
                )
                / 12.5
            )
            if self.replace_qu is not None:
                if self.replace_qu == "nothing":
                    sample[-1][1:, :, :] = torch.zeros_like(sample[-1][1:, :, :])
                    if self.print_first_time:
                        print("WARNING: replacing Q, U maps with zeros")
                        self.print_first_time = False
                if self.replace_qu == "t":
                    sample[-1][0, :, :] = torch.zeros_like(sample[-1][0, :, :])
                    if self.print_first_time:
                        print("WARNING: replacing T maps with zeros")
                        self.print_first_time = False
        else:
            sample.append(self._get_right_maps(self.input_type, map_idx, idx))
            sample.append(self._get_right_maps(self.output_type, map_idx, idx))

        if self.crop is not None:
            # Crop randomly
            sample = self._crop(sample[0], sample[1], npix=self.npix, crop=self.crop)

        if self.transform:
            sample[0] = self.transform(sample[0])

        return sample

    def _crop(self, *samples, npix, crop):
        """Randomly crops sample to a `crop` x `crop` image for every sample of size `npix` x `npix`"""
        x, y = random.randint(0, npix - crop), random.randint(0, npix - crop)
        samples = list(samples)
        for i in range(len(samples)):
            if isinstance(samples[i], list):
                for s in samples[i]:
                    if s.shape[-1] < 2:
                        continue
                    s = s[:, x : crop + x, y : crop + y]
            else:
                if samples[i].shape[-1] < 2:
                    continue
                samples[i] = samples[i][:, x : crop + x, y : crop + y]
        return list(samples)

    def _get_right_maps(self, map_types, map_idx, idx):
        """Return the right maps based on the content of map_types"""

        sample = []
        for map_type in map_types:
            # kappa map and mass
            if map_type == "kappa_map":
                sample.append(
                    torch.from_numpy(
                        self.kappa_maps[map_idx][self.maps[map_idx][idx][-1]][0]
                    ).float()[None, :]
                )
                continue

            if map_type == "mass":
                # Standardize maps
                sample.append(
                    (
                        torch.Tensor(
                            [self.kappa_maps[map_idx][self.maps[map_idx][idx][-1]][-1]]
                        ).float()[None, None, :]
                        - 12.5
                    )
                    / 12.5
                )
                continue

            # All QU maps
            if map_type == "qu":
                sample.append(torch.from_numpy(self.maps[map_idx][idx][0][1:]).float())

            # All TQU maps
            if map_type.endswith("maps"):
                if map_type.startswith("obs"):
                    sample.append(torch.from_numpy(self.maps[map_idx][idx][0]).float())
                elif map_type.startswith("len"):
                    sample.append(
                        torch.from_numpy(self.len_maps[map_idx][idx][0]).float()
                    )
                elif map_type.startswith("unl"):
                    sample.append(
                        torch.from_numpy(self.unl_maps[map_idx][idx][0]).float()
                    )
                elif map_type.startswith("dif"):
                    sample.append(
                        torch.from_numpy(
                            self.len_maps[map_idx][idx][0]
                            - self.unl_maps[map_idx][idx][0]
                        ).float()
                    )
                if self.replace_qu is not None:
                    if self.replace_qu == "nothing":
                        sample[-1][1:, :, :] = torch.zeros_like(sample[-1][1:, :, :])
                        if self.print_first_time:
                            print("WARNING: replacing Q, U maps with zeros")
                            self.print_first_time = False
                    elif self.replace_qu == "noise":
                        sample[-1][1:, :, :] = torch.randn(
                            size=sample[-1][1:, :, :].shape
                        )
                    elif self.replace_qu == "t":
                        sample[-1][0, :, :] = torch.zeros_like(sample[-1][0, :, :])
                        if self.print_first_time:
                            print("WARNING: replacing T maps with zeros")
                            self.print_first_time = False
                continue

            # Specific map
            j = list("tqu").index(map_type[4])
            if map_type.startswith("obs"):
                sample.append(
                    torch.from_numpy(self.maps[map_idx][idx][0][j]).float()[None, :]
                )
                continue
            elif map_type.startswith("len"):
                sample.append(
                    torch.from_numpy(self.len_maps[map_idx][idx][0][j]).float()[None, :]
                )
                continue
            elif map_type.startswith("unl"):
                sample.append(
                    torch.from_numpy(self.unl_maps[map_idx][idx][0][j]).float()[None, :]
                )
                continue
            elif map_type.startswith("dif"):
                sample.append(
                    torch.from_numpy(
                        self.len_maps[map_idx][idx][0][j]
                        - self.unl_maps[map_idx][idx][0][j]
                    ).float()[None, :]
                )

        if len(sample) == 1:
            return sample[0]
        return sample


class MapDataModule(pl.LightningDataModule):
    """Lensing Maps Data Module"""

    def __init__(
        self,
        train_dirs,
        val_dirs,
        batch_size,
        transform,
        num_workers,
        output_type,
        input_type,
        crop=None,
        replace_qu=None,
        ragh=False,
        **args
    ):
        super().__init__()
        self.train_dirs = train_dirs
        self.val_dirs = val_dirs
        self.batch_size = batch_size
        self.transform = transform
        self.num_workers = num_workers
        self.output_type = output_type
        self.input_type = input_type
        self.replace_qu = replace_qu
        self.ragh = ragh
        self.crop = crop
        self.masses_std = None
        self.masses_mean = None

    def setup(self, stage=None):
        if self.masses_std is None:
            # ADDED
            self.masses_std, self.masses_mean = 1, 0
        if stage == "fit" or stage is None:
            self.train_dataset = MapDataset(
                self.train_dirs,
                self.transform,
                self.output_type,
                input_type=self.input_type,
                crop=self.crop,
                replace_qu=self.replace_qu,
                masses_mean=self.masses_mean,
                masses_std=self.masses_std,
                ragh=self.ragh,
            )
            self.npix = self.train_dataset.npix

        if stage == "val" or stage is None:
            self.val_dataset = MapDataset(
                self.val_dirs,
                self.transform,
                self.output_type,
                input_type=self.input_type,
                crop=self.crop,
                replace_qu=self.replace_qu,
                masses_mean=self.masses_mean,
                masses_std=self.masses_std,
                ragh=self.ragh,
            )
            self.npix = self.val_dataset.npix

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
            num_workers=self.num_workers,
        )
