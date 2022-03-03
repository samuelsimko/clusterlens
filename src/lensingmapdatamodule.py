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
from map_parameters import MapParameters


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
        obj = lensingmap.cluster_maps(
            libdir,
            npix,
            lpix_amin,
            nsims_per_mass,
            results,
            {"M200c": M200, "z": z},
            profilename=profname,
            ellmax_sky=ellmaxsky,
        )
        kappa_map = obj.get_kappa_map(M200, z)
        datasets += [
            (obj.get_obs_map(idx, "t"), kappa_map) for idx in range(nsims_per_mass)
        ]

    return datasets


class LensingMapDataset(Dataset):
    """Lensing Maps Dataset"""

    def __init__(
        self,
        cambinifile,
        M200_list,
        nsims_per_mass,
        npix,
        lpix_amin,
        ellmaxsky,
        z,
        profname,
        results,
        transform=None,
    ):
        self.dataset = get_cluster_maps(
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

        sample = [torch.from_numpy(i).float()[None, :] for i in self.dataset[idx]]

        if self.transform:
            sample = self.transform(sample)

        return sample


class LensingMapDataModule(pl.LightningDataModule):
    """Lensing Maps Data Module"""

    def __init__(
        self,
        results,
        map_parameters: MapParameters,
        training_nsims_per_mass,
        validation_nsims_per_mass,
        batch_size,
        transform,
    ):
        super().__init__()
        self.map_parameters = map_parameters
        self.training_nsims_per_mass = training_nsims_per_mass
        self.validation_nsims_per_mass = validation_nsims_per_mass
        self.batch_size = batch_size
        self.transform = transform
        os.environ["LENSIT"] = TRAINING_LENSIT_DIR
        self.training_dataset = LensingMapDataset(
            **self.map_parameters.parameters,
            results=results,
            nsims_per_mass=self.training_nsims_per_mass,
            transform=self.transform,
        )
        os.environ["LENSIT"] = VALIDATION_LENSIT_DIR
        self.validation_dataset = LensingMapDataset(
            **self.map_parameters.parameters,
            results=results,
            nsims_per_mass=self.validation_nsims_per_mass,
            transform=self.transform,
        )

    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size)
