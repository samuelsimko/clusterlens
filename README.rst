===========
Clusterlens
===========

About
-----

This repository contains deep learning models to estimate the masses of galaxy clusters
from lensed CMB maps.

Features
--------

- Generation of training data using `LensIt <https://github.com/carronj/LensIt>`_
- MResUNet and ResNet models
- Training on T or TEB maps
- Two labels available: Kappa map or cluster mass.

Usage
-----

Create a virtual environment and activate it.
Install the requirements:

.. code-block:: console

   $ cd clusterlens
   $ pip install -r requirements.txt

Create simulated maps by executing `gen_maps.py`

.. code-block:: console

   $ python src/gen_maps.py 1 2 3 4 5 traindata --nsims 64 --cambinifile /path/to/cambinifile
   $ python src/gen_maps.py 1 2 3 4 5 validationdata --nsims 64 --cambinifile /path/to/cambinifile

Here, the script generates maps with masses in (1, 2, 3, 4) * 1e14 M☉.
`64` maps are created for each mass.
It will store the training maps in the `traindata` directory, and the validation maps in the 
`validationdata` directory.


To train a model, execute the `train.py` script.

.. code-block:: console

   $ python src/train.py --input_type obs_maps --output_type mass  --model mresunet --train_dir traindata --val_dir validationdata --batch_size 16 --max_epochs 30

For more information on the scripts, call them with the argument `--help`.

The training will create a log folder, which can be opened with Tensorboard.
If checkpointing is enabled, the trained model will be saved.
