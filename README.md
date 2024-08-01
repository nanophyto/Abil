

# Abil.py &middot; [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/nanophyto/Abil/LICENSE) [![Build Status](https://github.com/nanophyto/Abil/actions/workflows/ci.yml/badge.svg?branch=Continuous-integration)](https://github.com/nanophyto/Abil/actions/workflows/ci.yml?query=branch%3AContinuous-integration)

## Overview

Abil.py provides functions to interpolate distributions of biogeochemical observations using Machine Learning algorithms in Python. The library is optimized to interpolate many predictions in parallel and is thus particularly suited for distribution models of species, genes and transcripts. The library relies on [scikit-learn](https://scikit-learn.org/) and [MAPIE](https://mapie.readthedocs.io/).


Current support (v0.09):

- Random Forest, XGBoost, Bagged KNN, and MultiLayer Perceptrons

- Presence/Absence and continuous data

- 2-phase zero-inflated models

- Hyperparameter tuning and cross-validation

- Automatic feature scaling and one-hot-encoding

- Model prediction intervals (MAPIE)

- Example SLURM and Singularity scripts


Generally the workflow is as follows:

1. Define the model setup in a model_config.yml file (for an example see `/examples/configuration/`)
2. Tune the model for the species of interest using `tune.py`
3. Predict the distribution of each species using `predict.py`
4. Merge the predictions into a single netcdf and do post processing using `post.py`

Examples for each step are provided in the respective Jupyter notebooks which can be found in `/examples/notebooks`.

## Directory structure

The recommended directory structure is:

```bash

Abil
├── abil
|   └── __init__.py
|   └── functions.py
|   └── post.py
|   └── predict.py
|   └── tune.py
├── dist
|   └── abil-0.0.9.tar.gz
|   └── abil-0.0.9-py3-none-any.whl
├── docs
├── examples
|   └── conda
|       └── environment.yml
|   └── configuration
|       └── 2-phase.yml
|       └── classifier.yml
|       └── regressor.yml
|   └── data
|       └── prediction.csv
|       └── targets.csv
|       └── training.csv
|   └── hpc_example
|       └── hpc_post.py
|       └── hpc_predict.py
|       └── hpc_tune.py
|       └── post.sh
|       └── predict.sh
|       └── README.md
|       └── tune_KNN.sh
|       └── tune_RF.sh
|       └── tune_XGB.sh
|   └── notebooks
|       └── tune.ipynb
|       └── predict.ipynb
|       └── post.ipynb
|   └── singularity
|       └── singularity.sif
├── studies
|   └── devries2024
├── tests
├── README.md
├── pyproject.toml
└── README.md

```

## Installing the package:

Install the dependencies in a new environment: 

``` conda env create -f package_save_path/examples/conda/environment.yml ``` 

Activate the new environment and install Abil:

``` conda activate abil-env ``` 

``` python -m pip install package_save_path/Abil/dist/abil-0.0.9.tar.gz  ``` 

## Updating the package:

If you have changed the scripts and want to update the package, a new version can be build.

CD to the planktonSDM directory, then run:

``` python3 -m build  ``` 

Note: if you want to change the version name of the package, this can be changed in:

`pyproject.toml`

## Running the model on a hpc cluster

See: `/examples/hpc_example/README.md`