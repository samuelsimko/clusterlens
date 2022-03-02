# -*- coding: utf-8 -*-
import numpy as np

# Parameters for data generation

gupta_training_data_params = {
    # M200 masses of generated maps
    "M200_list": 10e14
    * np.concatenate(
        (np.linspace(1, 10, 10), np.linspace(20, 100, 9), np.linspace(200, 500, 4))
    ),
    # Number of generated maps
    "nsims_per_mass": 400,
    # Number of pixels in row/column
    "npix": 64,
    # Physical size of a pixel in arcmin
    "lpix_amin": 0.3,
    # Maximum multipole used to generate the CMB maps from the CMB power spectra
    "ellmaxsky": 6000,
    # Redshift
    "z": 0.7,
    # Profile name
    "profname": "nfw",
}

gupta_validation_data_params = gupta_training_data_params.copy()
gupta_validation_data_params.update({"nsims_per_mass": 200})

small_data_params = {
    # M200 masses of generated maps
    "M200_list": [1e14, 2e14, 3e14, 4e14],
    # Number of generated maps
    "nsims_per_mass": 25,
    # Number of pixels in row/column
    "npix": 64,
    # Physical size of a pixel in arcmin
    "lpix_amin": 0.3,
    # Maximum multipole used to generate the CMB maps from the CMB power spectra
    "ellmaxsky": 6000,
    # Redshift
    "z": 0.7,
    # Profile name
    "profname": "nfw",
}
