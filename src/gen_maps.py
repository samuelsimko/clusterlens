# -*- coding: utf-8 -*-
import os
import numpy as np
import pickle


import camb
import click

from lensit.clusterlens import lensingmap
from tqdm import tqdm


def initialize_camb(cambinifile):
    print("Initializing CAMB... ", end="", flush=True)
    pars = camb.read_ini(cambinifile)
    print("Getting Results... ", end="", flush=True)
    results = camb.get_results(pars)
    print("Done", flush=True)
    return results


def get_cluster_maps(
    cambinifile, results, nsims, npix, lpix_amin, ellmaxsky, mass, z, profname, **kwargs
):

    kappa_maps = []
    maps = []
    print(kwargs)

    for i, M200 in enumerate(tqdm(mass)):
        libdir = lensingmap.get_cluster_libdir(
            cambinifile, profname, npix, lpix_amin, ellmaxsky, M200, z, nsims
        )
        print("Using libdir {}".format(libdir))
        obj = lensingmap.cluster_maps(
            libdir=libdir,
            npix=npix,
            lpix_amin=lpix_amin,
            nsims=nsims,
            cosmo=results,
            profparams={"M200c": M200 * 10e14, "z": z},
            profilename=profname,
            ellmax_sky=ellmaxsky,
        )
        kappa_maps += [[obj.get_kappa_map(M200 * 10e14, z), M200]]
        maps += [
            [*(obj.get_obs_map(idx, f).astype(float) for f in ["t", "q", "u"]), i]
            for idx in range(nsims)
        ]

    return np.array(maps), np.array(kappa_maps)


@click.command()
@click.argument("mass", nargs=-1, type=int)
@click.argument("destdir", nargs=1, type=click.Path(exists=False))
@click.option(
    "--nsims", help="Number of simulations for each mass", required=True, type=int
)
@click.option(
    "--npix",
    default=64,
    show_default=True,
    help="Number of pixels in each line/column of the map.",
)
@click.option(
    "--ellmaxsky",
    default=6000,
    show_default=True,
    help="Maximum multipole used to generate the CMB maps from the CMB power spectra",
)
@click.option(
    "--lpix_amin",
    default=0.3,
    show_default=True,
    help="Physical size of a pixel in arcmin",
)
@click.option("--z", default=0.7, show_default=True, help="Redshift")
@click.option(
    "--profname",
    default="nfw",
    show_default=True,
    help="Mass model profile name (currently only supports nfw)",
    type=click.Choice(["nfw"], case_sensitive=False),
)
@click.option(
    "--cambinifile",
    help="CAMB initialization file to use",
    required=True,
    type=click.Path(exists=True),
)
def genmaps(**args):
    """A program to generate CMB lensed maps."""
    args["destdir"] = os.path.join(os.getcwd(), args["destdir"])
    results = initialize_camb(args["cambinifile"])
    os.environ["LENSIT"] = args["destdir"]
    maps, kappa_maps = get_cluster_maps(results=results, **args)
    np.save(os.path.join(args["destdir"], "maps"), maps, allow_pickle=True)
    np.save(os.path.join(args["destdir"], "kappa_maps"), kappa_maps, allow_pickle=True)
    pickle.dump(args, open(os.path.join(args["destdir"], "args"), "wb"))


if __name__ == "__main__":
    genmaps()
