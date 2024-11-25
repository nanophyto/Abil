#!/bin/bash
#
#
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --mem=60000M
#SBATCH --cpus-per-task=1
#SBATCH --account=GEOG024542

module  load apptainer/1.1.9 openmpi/4.1.2 

srun singularity exec \
-B/user/work/$(whoami):/user/work/$(whoami) \
/user/work/$(whoami)/Abil/singularity/abil.sif \
python /user/work/$(whoami)/Abil/studies/wiseman2024/hpc_post.py 

export SINGULARITY_CACHEDIR=/user/work/$(whoami)/.singularity
