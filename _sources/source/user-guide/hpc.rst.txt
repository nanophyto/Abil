High Performance Computing
==========================

Prerequisites
-------------


Singularity
-----------
To run abil on a HPC machine, first compile singularity from the terminal:

First, cd to the singularity folder:

.. code-block:: 

    cd ./Abil/examples/singularity/

If using apptainer (recommended):

.. code-block:: 

    sudo apptainer build abil.sif Singularity.sif

If using singularity:

.. code-block:: 

    sudo singularity build abil.sif Singularity.sif

Note: apptainer is often easier to install than singularity and is backwards compatible with legacy Singularity installs:

`Install Apptainer <https://apptainer.org/docs/admin/main/installation.html>`_

`Install Singularity <https://apptainer.org/docs/admin/1.2/installation.html>`_

Transfer Abil to your HPC Machine
---------------------------------

.. tab-set::

    .. tab-item:: Unix/MacOS
        
        To transfer to your home directory (~):

        .. code-block:: 

           scp <./Abil> <username@HPC_machine.ac.uk:~> 

        To transfer to a specific directory (ex. /user/work/username):

        .. code-block:: 

         scp <./Abil> <username@HPC_machine.ac.uk:/user/work/username>


    .. tab-item:: Windows

        To transfer from a Windows machine, use `WinSCP <https://winscp.net/eng/index.php>`_.

        To use WinSCP, type the host in the `Host name` box, then enture your username in the `User Name` box.

        For more instructions, check with your organization.

Execute Abil on your HPC Machine
--------------------------------

To login to your HPC account, follow your organizations directions.

Example (ssh):

.. code-block:: 

    ssh username@HPC_machine.ac.uk

Change directory to abil:

.. code-block:: 

    cd /user/work/username/Abil

Change directory to the folder containing your bash scripts:

.. code-block:: 

    cd hpc_example

Submit hpc tuning jobs (ex. using slurm):

.. code-block:: 

    sbatch tune_RF.sh
    sbatch tune_KNN.sh
    sbatch tune_XGB.sh

After tune.sh jobs are finshed, submit predict job:

.. code-block:: 

    sbatch predict.sh

After predict.sh is finished, submit post job:

.. code-block:: 

    sbatch post.sh

Singularity file
----------------
Below is the Singularity.sif file text. This is used to create abil.sif in the steps above.

.. code-block:: singularity

    Bootstrap: docker
    From: continuumio/miniconda3

    %files
        ../../dist/abil-0.0.10.tar.gz /root
        ../../examples/conda/environment.yml /root

    %post
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
        . /opt/conda/etc/profile.d/conda.sh
        conda install -n base conda-libmamba-solver
        conda config --set solver libmamba
        conda config --set channel_priority true
        conda config --add channels conda-forge
        conda env update -n base --file /root/environment.yml
        cd
        python -m pip install abil-0.0.10.tar.gz

    %runscript
        . /opt/conda/etc/profile.d/conda.sh
        exec "$@"