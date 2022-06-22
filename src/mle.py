import numpy as np
import os
import tqdm
import sys
import click
import random


def mle(dirmaps, dirtest, dirdest, map_list):
    """Predict likelihoods for one covariance matrix"""

    maps_test = np.load(os.path.join(dirtest, "maps.npy"), allow_pickle=True)

    save_mass = True

    masses = np.load(os.path.join(dirtest, "masses.npy"), allow_pickle=True)

    for idx, path_data in enumerate(dirmaps):

        t = []
        qu = []

        maps_data = np.load(os.path.join(path_data, "maps.npy"), allow_pickle=True)

        print(maps_data.shape[0], "Dataset size")
        for i in range(maps_data.shape[0]):
            t.append(maps_data[i][0, :, :].flatten())
            qu.append(maps_data[i][1:, :, :].flatten())

        def get_sigma_lens(t):
            t = np.array([ti.flatten() for ti in t])
            s = np.cov(t, rowvar=False)
            print("inverting...")
            return s, np.linalg.inv(s)

        st, stinv = get_sigma_lens(t)
        squ, squinv = get_sigma_lens(qu)

        ln_det_st = np.linalg.slogdet(st)
        print("rank: ", np.linalg.matrix_rank(st))
        print("Det: ", ln_det_st)

        ln_det_qu = np.linalg.slogdet(squ)
        print("rank: ", np.linalg.matrix_rank(squ))
        print("Det: ", ln_det_qu)

        def get_likelihood(d, sinv, ln_det):
            d = d[None, :]
            return np.abs(d @ sinv @ d.T)  # + ln_det[1]

        tres = []
        qures = []

        for i in tqdm.tqdm(range(maps_test.shape[0])):
            ti = (maps_test[i][0, :, :]).flatten()
            qui = (maps_test[i][1:, :, :]).flatten()
            tres.append(get_likelihood(ti, stinv, ln_det_st).item())
            qures.append(get_likelihood(qui, squinv, ln_det_qu).item())

        np.savetxt(
            os.path.join(dirdest, "tres_{}".format(map_list[idx])), np.array(tres)
        )
        np.savetxt(
            os.path.join(dirdest, "qures_{}".format(map_list[idx])), np.array(qures)
        )

        if save_mass:
            save_mass = False  # Only save the masses once
            np.savetxt(os.path.join(dirdest, "masses"), np.array(masses))


@click.command()
@click.argument("map_list", nargs=-1, type=float)
@click.option("dirtest", nargs=1, type=click.Path(exists=True), required=True)
@click.option("dirdest", nargs=1, type=click.Path(exists=False), required=True)
@click.option("dirmaps", nargs=1, type=click.Path(exists=True), required=True)
def mlecli(**args):
    mle(args["dirmaps"], args["dirtest"], args["dirdest"], args["map_list"])


if __name__ == "__main__":
    mlecli()
