.. _getting-started:

============
Installation
============

Prerequisites
-------------

Ensure you have the following installed on your system:

- `Python <https://www.python.org/downloads/>`_ (>=3.7 required)
- `Git <https://git-scm.com/downloads>`_
- `pip <https://pip.pypa.io/en/stable/installation/>`_

Install via pip
---------------

Run the following command to install the package directly from GitHub:

.. code-block:: sh

   pip install abil

.. note::
   If you are using a conda environment remember to activate it before running pip

Install via Cloning (for Development)
-------------------------------------

If you want to modify the package, clone the repository and install it using `PIP editable install <https://setuptools.pypa.io/en/latest/userguide/development_mode.html>`_:

.. code-block:: sh

   git clone https://github.com/nanophyto/Abil.git
   cd Abil
   pip install -e .