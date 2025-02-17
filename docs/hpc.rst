High Performance Computing
==========================

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