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

Using Anaconda
---------------
When using Python it is strongly recommended to use `Anaconda <https://docs.anaconda.com/miniconda/>`_ environments to avoid package conflicts.



If you decide to use Anaconda, an Abil specific environment can be created as below:

.. tab-set::

    .. tab-item:: Unix/MacOS
        
        To create a new Conda environment and install the dependencies from ``Abil/requirements.txt``, run the following commands:

        .. code-block:: sh

            # Create a new Conda environment (replace 'myenv' with your desired environment name)
            conda create --name myenv 

            # Activate the environment
            conda activate myenv

            # Install the dependencies from requirements.txt
            pip install -r /path/to/requirements.txt

    .. tab-item:: Windows
        
        To create a new Conda environment and install the dependencies from ``Abil\requirements.txt``, run the following commands in Command Prompt or Anaconda Prompt:

        .. code-block:: bat

            # Create a new Conda environment (replace 'myenv' with your desired environment name)
            conda create --name myenv

            # Activate the environment
            conda activate myenv

            # Install the dependencies from requirements.txt
            pip install -r C:\path\to\requirements.txt

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