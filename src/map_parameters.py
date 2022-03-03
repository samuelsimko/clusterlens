# -*- coding: utf-8 -*-
import os
import os.path as op

import camb
import numpy as np

from directories import CAMB_INI_DIR

# Parameters for data generation


def initialize_camb(cambinifile):
    os.environ["CAMBINIDIR"] = CAMB_INI_DIR
    pars = camb.read_ini(op.join(os.getenv("CAMBINIDIR"), cambinifile + ".ini"))
    return camb.get_results(pars)


class MapParameters:
    def __init__(self, cambinifile, M200_list, npix, lpix_amin, ellmaxsky, z, profname):
        self.parameters = {
            "cambinifile": cambinifile,
            "M200_list": M200_list,
            "npix": npix,
            "lpix_amin": lpix_amin,
            "ellmaxsky": ellmaxsky,
            "z": z,
            "profname": profname,
        }
