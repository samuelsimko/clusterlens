import numpy as np

from map_parameters import MapParameters

gupta_map_parameters = MapParameters(
    cambinifile="planck_2018_acc",
    M200_list=10e14 * np.concatenate(
        (np.linspace(1, 10, 10), np.linspace(20, 100, 9), np.linspace(200, 500, 4))
    ),
    npix=64,
    lpix_amin=0.3,
    ellmaxsky=6000,
    z=0.7,
    profname="nfw"
)
