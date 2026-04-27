

# Abil.py &middot; 
[![DOI](https://joss.theoj.org/papers/10.21105/joss.08755/status.svg)](https://doi.org/10.21105/joss.08755) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16886567.svg)](https://doi.org/10.5281/zenodo.16886567) [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/nanophyto/Abil/LICENSE) [![Build Status](https://github.com/nanophyto/Abil/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/nanophyto/Abil/actions/workflows/ci.yml?query=branch%3Amain)  [![dev](https://img.shields.io/badge/docs-Dev_Docs-blue)](https://nanophyto.github.io/Abil/)

## Overview

Abil.py provides functions to interpolate distributions of biogeochemical observations using Machine Learning algorithms in Python. The library is optimized to interpolate many predictions in parallel and is thus particularly suited for distribution models of species, genes and transcripts. The library relies on [scikit-learn](https://scikit-learn.org/).

## Installation

### Prerequisites
Ensure you have the following installed on your system:
- [Python](https://www.python.org/downloads/) (>=3.7, <3.13 required)
- [Git](https://git-scm.com/downloads)
- [pip](https://pip.pypa.io/en/stable/installation/)

### Using Anaconda 
We recommend uing [Anaconda](https://www.anaconda.com/download) environments to avoid package conflicts.

Run the following command to install the package with Anaconda:

```sh
conda create -n myenv python=3.11 pip
conda activate myenv
pip install abil
```

### Install via pip
Run the following command to install the package directly:
```sh
pip install abil
```

### Install via cloning (for development)
If you want to modify the package, clone the repository and install it in editable mode:
```sh
git clone https://github.com/nanophyto/Abil.git
cd Abil
pip install -e .
```

### Run unit test
To run a unit test, make sure you are under the project root:
```sh
python -m unittest tests/test.py
```


## Documentation

See the [documentation](https://nanophyto.github.io/Abil/) for instructions on how to setup and run the models.
