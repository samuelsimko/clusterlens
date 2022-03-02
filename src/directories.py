# -*- coding: utf-8 -*-
import os.path as op
import camb

# This file must contain the path to directories for several tasks

# Directory to store the training data
TRAINING_LENSIT_DIR = "/home/sam/unige/bachelor_project/clusterlens/data/train"

# Directory to store the validation data
VALIDATION_LENSIT_DIR = "/home/sam/unige/bachelor_project/clusterlens/data/validation"

# Path to the CAMB inifiles directory
# CAMB_INI_DIR = op.join(op.dirname(camb.__path__[0]),  'inifiles')
CAMB_INI_DIR = "/home/sam/unige/bachelor_project/CAMB/inifiles"
