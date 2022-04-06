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
    ):
        super().__init__()

        self.transform = transform
        self.output_type = output_type
        self.input_type = input_type
        self.crop = crop

        self.args = []
        self.maps = []
        self.kappa_maps = []
        self.unl_maps = []
        self.len_maps = []

        for path in paths:
            self.args.append(pickle.load(open(os.path.join(path, "args"), "rb")))
            self.maps.append(np.load(os.path.join(path, "maps.npy"), allow_pickle=True))
            self.kappa_maps.append(
                np.load(os.path.join(path, "kappa_maps.npy"), allow_pickle=True)
            )
            self.unl_maps.append(
                np.load(os.path.join(path, "unl_maps.npy"), allow_pickle=True)
            )
            self.len_maps.append(
                np.load(os.path.join(path, "len_maps.npy"), allow_pickle=True)
            )

        self.npix = self.args[0]["npix"]

        # Get cumulated sum of the number of maps in each path - 1
        self.len_cumsum = (
            np.cumsum([len(arg["mass"]) * arg["nsims"] for arg in self.args]) - 1
        )
        self.len = self.len_cumsum[-1] + 1

        # Get masses
        self.masses = np.sort(
            np.unique(np.array([km[:, -1] for km in self.kappa_maps]).flatten())
        ).astype(float)
        self.masses = np.log(self.masses / 500)
        self.masses_std, self.masses_mean = np.std(self.masses), np.mean(self.masses)
        self.mass_normalize = Normalize(self.masses_mean, self.masses_std)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        # Get index to use within self.maps
        map_idx = np.searchsorted(self.len_cumsum, idx)
        if map_idx != 0:
            idx -= self.len_cumsum[map_idx - 1] + 1

        sample = []

        sample.append(self._get_right_maps(self.input_type, map_idx, idx))
        sample.append(self._get_right_maps(self.output_type, map_idx, idx))

        if self.crop is not None:
            # Crop randomly
            sample = self._crop(sample[0], sample[1], npix=self.npix, crop=self.crop)

        if self.transform:
            sample[0] = self.transform(sample[0])
            # sample[1] = self.transform(sample[1])

        return sample

    def _crop(self, *samples, npix, crop):
        """Randomly crops sample to a `crop` x `crop` image for every sample of size `npix` x `npix`"""
        # x, y = random.randint(0, npix - crop), random.randint(0, npix - crop)
        x, y = (npix - crop) // 2, (npix - crop) // 2
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
                    self.mass_normalize(
                        torch.log(
                            torch.Tensor(
                                [
                                    self.kappa_maps[map_idx][
                                        self.maps[map_idx][idx][-1]
                                    ][-1]
                                ]
                            ).float()[None, None, :]
                            / 500
                        )
                    )
                )
                continue

            # All TQU maps
            if map_type.endswith("maps"):
                if map_type.startswith("obs"):
                    sample.append(torch.from_numpy(self.maps[map_idx][idx][0]).float())
                    continue
                elif map_type.startswith("len"):
                    sample.append(
                        torch.from_numpy(self.len_maps[map_idx][idx][0]).float()
                    )
                    continue
                elif map_type.startswith("unl"):
                    sample.append(
                        torch.from_numpy(self.unl_maps[map_idx][idx][0]).float()
                    )
                    continue
                elif map_type.startswith("dif"):
                    sample.append(
                        torch.from_numpy(
                            self.len_maps[map_idx][idx][0]
                            - self.unl_maps[map_idx][idx][0]
                        ).float()
                    )
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
        self.crop = crop

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = MapDataset(
                self.train_dirs,
                self.transform,
                self.output_type,
                input_type=self.input_type,
                crop=self.crop,
            )
            self.npix = self.train_dataset.npix
            self.masses = self.train_dataset.masses

        if stage == "val" or stage is None:
            self.val_dataset = MapDataset(
                self.val_dirs,
                self.transform,
                self.output_type,
                input_type=self.input_type,
                crop=self.crop,
            )
            self.npix = self.val_dataset.npix
            self.masses = self.val_dataset.masses

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
