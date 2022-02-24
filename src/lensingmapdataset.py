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
    nsims,
    npix,
    lpix_amin,
    ellmaxsky,
    M200_range,
    z,
    profname="nfw",
):

    assert (
        nsims % len(M200_range) == 0
    ), "nsims must be dividable by the number of different M200"

    nsims_per_mass = nsims // len(M200_range)
    datasets = []

    for M200 in M200_range:
        libdir = lensingmap.get_cluster_libdir(
            cambinifile, profname, npix, lpix_amin, ellmaxsky, M200, z, nsims
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
        npix,
        nsims,
        lpix_amin,
        profname,
        ellmaxsky,
        M200_range,
        z,
        transform=None,
    ):
        self.datasets = get_cluster_maps(
            cambinifile,
            results,
            nsims,
            npix,
            lpix_amin,
            ellmaxsky,
            M200_range,
            z,
            profname=profname,
        )
        self.transform = transform
        self.npix = npix
        self.nsims = nsims
        self.z = z
        self.M200_range = M200_range
        self.nmass = len(M200_range)
        self.nsims_per_mass = nsims / len(M200_range)

    def __len__(self):
        return self.nsims

    def __getitem__(self, idx):
        i = int(idx // self.nsims_per_mass)
        x = self.datasets[i]
        sample = torch.from_numpy(
            x.get_unl_map(idx % self.nsims_per_mass, "t")
        ), torch.from_numpy(x.get_kappa_map(self.M200_range[i], self.z))

        if self.transform:
            sample = self.transform(sample)

        return sample
