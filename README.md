

# Abil.py &middot; [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/nanophyto/Abil/LICENSE) [![Build Status](https://github.com/nanophyto/Abil/actions/workflows/ci.yml/badge.svg?branch=Continuous-integration)](https://github.com/nanophyto/Abil/actions/workflows/ci.yml?query=branch%3AContinuous-integration) [![dev](https://img.shields.io/badge/docs-Dev_Docs-blue)](https://nanophyto.github.io/Abil/)

## Overview

Abil.py provides functions to interpolate distributions of biogeochemical observations using Machine Learning algorithms in Python. The library is optimized to interpolate many predictions in parallel and is thus particularly suited for distribution models of species, genes and transcripts. The library relies on [scikit-learn](https://scikit-learn.org/).

## Installation

### Prerequisites
Ensure you have the following installed on your system:
- [Python](https://www.python.org/downloads/) (>=3.7 recommended)
- [Git](https://git-scm.com/downloads)
- [pip](https://pip.pypa.io/en/stable/installation/)

### Install via pip
Run the following command to install the package directly from GitHub:
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
