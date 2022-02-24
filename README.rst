===========
clusterlens
===========

About
-----

This repository contains deep learning models to estimate the masses of galaxy clusters
using weak gravitational CMB lensing.

Usage
-----

Create a virtual environment and activate it.
Then, install the requirements:

.. code-block:: console

   $ cd clusterlens
   $ pip install -r requirements.txt

Change the LENSIT environment variable to somewhere safe to write (for example,
clusterlens/pwd/data/cached)

.. code-block:: console

   $ export LENSIT=$(pwd)/data/cached

Change the CAMBINIDIR environment variable to the path of your CAMB `inifiles` folder

.. code-block:: console

   $ export CAMBINIDIR=/path/to/your/inifiles

You can now train the MResUNet on small simulated data by running a training script.

.. code-block:: console

   $ python src/train.py
