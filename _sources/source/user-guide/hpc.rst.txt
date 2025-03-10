High Performance Computing
==========================
.. note::

    This tutorial assumes some basic familiarity with Unix and high performance computing.
    To run the scripts on your HPC system, it needs to have both SLURM and Apptainer or Singularity installed.
    If you are unsure this is the case, we recommend contacting your HPC admin team.


HPC at the University of Bristol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you are running Abil at the University of Bristol, we recommend first completing the `Getting Started HPC Course <https://github.com/UoB-HPC/hpc-course-getting-started>`_.
If you are running at another institution, we recommend familiarizing yourself with your local HPC machine.

Installing Singularity
~~~~~~~~~~~~~~~~~~~~~~
To simplify installing packages and dependencies on HPC machines, which often require admin approval, we use containers.
Here we use Singularity, which packages all of the requirements for running Abil into a portable and reproducible container that does not require root privileges.
There are two software options for creating Singularity containers, Singularity and Apptainer.
Apptainer is often easier to install than Singularity and is backwards compatible with legacy Singularity installs.
Both require a Linux operating system, but provide instructions for installing on Windows or Mac OS.

.. tab-set::

    .. tab-item:: Unix

        `Install Apptainer <https://apptainer.org/docs/admin/main/installation.html>`_

        `Install Singularity <https://docs.sylabs.io/guides/3.0/user-guide/installation.html>`_

    .. tab-item:: Windows/MacOS

        Singularity/Apptainer is not natively supported on Windows or MacOS and requires a Virtual Machine (VM).
    	For more details see the `Apptainer admin guide <https://apptainer.org/docs/admin/main/installation.html#installation-on-windows-or-mac>`_
        or the `Singularity admin guide <https://docs.sylabs.io/guides/3.0/user-guide/installation.html#install-on-windows-or-mac>`_




Building Singularity Container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run Abil on a HPC machine, first compile Singularity from the terminal:

First, change directory to the Singularity folder:

.. code-block:: 

    cd ./Abil/singularity/

If using Apptainer (recommended):

.. code-block:: 

    sudo apptainer build abil.sif Singularity.sif

If using Singularity:

.. code-block:: 

    sudo singularity build abil.sif Singularity.sif

Transfer Abil to your HPC Machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::

    .. tab-item:: Unix/MacOS
        
        To transfer to your home directory (~):

        .. code-block:: 

           scp -r ./Abil <username>@HPC_machine.ac.uk:~ 

        To transfer to a specific directory (ex. /user/work/username):

        .. code-block:: 

            scp -r ./Abil <username>@HPC_machine.ac.uk:/user/work/username


    .. tab-item:: Windows

        To transfer from a Windows machine, use `WinSCP <https://winscp.net/eng/index.php>`_.

        To use WinSCP, type the host in the `Host name` box, then enture your username in the `User Name` box.

        For more instructions, check with your organization.

SLURM scripts
~~~~~~~~~~~~~

To execute Abil on an HPC machine, we use SLURM scripts. The SLURM script tells the HPC machine what to load (the Singularity container), what to execute (Python scripts), and how much compute is required, in a single executable file.


Variable declarations
^^^^^^^^^^^^^^^^^^^^^
The first part of the bash script declares the variables needed to execute the job.
Here, we include the time limit for the run (time), the number of nodes to use (nodes), the memory allocation (mem),
the number of cpus per task (cpus-per-task), and the number of targets to be tuned (array).

.. literalinclude:: ../../examples/tune_KNN.sh
    :lines: 1-8
    :language: shell
.. note::

    The wall time (`--time`), amount of RAM (`--mem`) and number of threads (`--cpus-per-task`) will vary depending on the size of your dataset, the number of hyper-parameters and your HPC hardware.

Executable commands
^^^^^^^^^^^^^^^^^^^
The next part of the bash script includes the commands to be executed.
First, the array value is used to set a local variable that will be used to specify the target being tuned.

.. literalinclude:: ../../examples/tune_KNN.sh
    :lines: 10
    :language: shell

Next, the Apptainer module is loaded, and set up using the abil.sif container uploaded prior.

.. literalinclude:: ../../examples/tune_KNN.sh
    :lines: 12-16
    :language: shell

Finally, the model Python script is executed using the specified number of cpus, for the target "i", within a specific model (knn in this instance).
Lastly, the Singularity cache is exported.

.. literalinclude:: ../../examples/tune_KNN.sh
    :lines: 17-19
    :language: shell

Alterations for predict and post
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The set up is the same for each the predict.sh and post.sh scripts, with the only change being the Python executable line.
predict.sh should say the following:

.. literalinclude:: ../../examples/predict.sh
    :lines: 17
    :language: shell

while post.sh should say the following, and does not include the array specification:

.. literalinclude:: ../../examples/post.sh
    :lines: 14
    :language: shell

Execute Abil on your HPC Machine
--------------------------------

To login to your HPC account, follow your organizations directions.

Example (ssh):

.. code-block:: 

    ssh <username>@HPC_machine.ac.uk

Change directory to Abil:

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

.. literalinclude:: ../../examples/Singularity.sif
    :language: Singularity