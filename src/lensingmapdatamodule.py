# -*- coding: utf-8 -*-
import os
import os.path as op

import camb
import torch
import pytorch_lightning as pl

from os import getenv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from lensit.clusterlens import lensingmap

from directories import TRAINING_LENSIT_DIR, VALIDATION_LENSIT_DIR
# from data_parameters import gupta_map_parameters
from data_parameters import MapParameters

def initialize_camb(cambinifile):
    print("Initializing CAMB...", end="")
    pars = camb.read_ini(op.join(getenv("CAMBINIDIR"), cambinifile + ".ini"))
    print("Done")
    return camb.get_results(pars)

def get_cluster_maps(
    cambinifile,
    results,
    nsims_per_mass,
    npix,
    lpix_amin,
    ellmaxsky,
    M200_list,
    z,
    profname="nfw",
):

    datasets = []

    for M200 in M200_list:
        libdir = lensingmap.get_cluster_libdir(
            cambinifile, profname, npix, lpix_amin, ellmaxsky, M200, z, nsims_per_mass
        )
        print("Using libdir {}".format(libdir))
        datasets.append(
            lensingmap.cluster_maps(
                libdir,
                npix,
                lpix_amin,
                nsims_per_mass,
                results,
                {"M200c": M200, "z": z},
                profilename=profname,
                ellmax_sky=ellmaxsky,
            )
        )

    return datasets


class LensingMapDataset(Dataset):
    """Lensing Maps Dataset"""

    def __init__(
        self,
        cambinifile,
        results,
        M200_list,
        nsims_per_mass,
        npix,
        lpix_amin,
        ellmaxsky,
        z,
        profname,
        transform=None,
    ):
        self.datasets = get_cluster_maps(
            cambinifile,
            results,
            nsims_per_mass,
            npix,
            lpix_amin,
            ellmaxsky,
            M200_list,
            z,
            profname=profname,
        )
        self.transform = transform
        self.npix = npix
        self.z = z
        self.M200_list = M200_list
        self.nmass = len(M200_list)
        self.nsims_per_mass = nsims_per_mass
        self.nsims = nsims_per_mass * self.nmass

    def __len__(self):
        return self.nsims

    def __getitem__(self, idx):
        i = int(idx // self.nsims_per_mass)
        x = self.datasets[i]
        sample = [torch.from_numpy(
            x.get_obs_map(idx % self.nsims_per_mass, "t")
        ), torch.from_numpy(x.get_kappa_map(self.M200_list[i], self.z))]

        if self.transform:
            sample = self.transform(sample)

        return sample


class LensingMapDataModule(pl.LightningDataModule):
    """Lensing Maps Data Module"""

    def __init__(
        self,
        map_parameters: MapParameters,
        training_nsims_per_mass,
        validation_nsims_per_mass,
        batch_size,
        transform
    ):
        super().__init__()
        self.map_parameters = map_parameters
        self.training_nsims_per_mass = training_nsims_per_mass
        self.validation_nsims_per_mass = validation_nsims_per_mass
        self.batch_size = batch_size
        self.transform = transform

    def setup(self, stage=None):

        # Assign Train/val split(s) for use in Dataloaders
        if stage in (None, "fit"):
            os.environ["LENSIT"] = TRAINING_LENSIT_DIR
            self.training_dataset = LensingMapDataset(
                **self.map_parameters.parameters,
                nsims_per_mass=self.training_nsims_per_mass,
                transform=self.transform,
            )
            os.environ["LENSIT"] = VALIDATION_LENSIT_DIR
            self.validation_dataset = LensingMapDataset(
                **self.map_parameters.parameters,
                nsims_per_mass=self.validation_nsims_per_mass,
                transform=self.transform,
            )


    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size)
