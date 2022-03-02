# -*- coding: utf-8 -*-
import os.path as op

import camb
import torch

from os import getenv
from torch.utils.data import Dataset
from lensit.clusterlens import lensingmap


def initialize_camb(cambinifile):
    pars = camb.read_ini(op.join(getenv("CAMBINIDIR"), cambinifile + ".ini"))
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
        sample = torch.from_numpy(
            x.get_obs_map(idx % self.nsims_per_mass, "t")
        ), torch.from_numpy(x.get_kappa_map(self.M200_list[i], self.z))

        if self.transform:
            sample = self.transform(sample)

        return sample
