Abil.py
=======================================
*Aquatic Biogeochemical Interpolation Library in Python*

Abil.py is a Python library to interpolate distributions of biogeochemical observations using Machine Learning algorithms in Python. 
The library is optimized to interpolate many predictions in parallel and is thus particularly suited for distribution models of species, 
genes and transcripts. The library relies on scikit-learn and MAPIE.

Quick install
***********************

Install the dependencies in a new environment:

>>> conda env create -f package_save_path/examples/conda/environment.yml

Activate the new environment:

>>> conda activate abil-env

Install Abil:

>>> python -m pip install package_save_path/Abil/dist/abil-0.0.9.tar.gz 

.. toctree::
   index
   quick-start
   examples
   yaml-config
   mapie
   hpc

