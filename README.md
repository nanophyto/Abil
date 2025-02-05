

# Abil.py &middot; [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/nanophyto/Abil/LICENSE) [![Build Status](https://github.com/nanophyto/Abil/actions/workflows/ci.yml/badge.svg?branch=Continuous-integration)](https://github.com/nanophyto/Abil/actions/workflows/ci.yml?query=branch%3AContinuous-integration)[![Dev Docs](https://img.shields.io/badge/docs-Dev_Docs-blue)](https://nanophyto.github.io/Abil/)

## Overview

Abil.py provides functions to interpolate distributions of biogeochemical observations using Machine Learning algorithms in Python. The library is optimized to interpolate many predictions in parallel and is thus particularly suited for distribution models of species, genes and transcripts. The library relies on [scikit-learn](https://scikit-learn.org/).

## Documentation

For detailed developer documentation, please visit [https://nanophyto.github.io/Abil/](https://nanophyto.github.io/Abil/).


## Installing the package:

Install the dependencies in a new environment: 

``` conda env create -f package_save_path/examples/conda/environment.yml ``` 

Activate the new environment and install Abil:

``` conda activate abil-env ``` 

``` python -m pip install package_save_path/Abil/dist/abil-0.0.9.tar.gz  ``` 