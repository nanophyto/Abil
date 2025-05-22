#!/bin/bash
#
#
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --mem=60000M
#SBATCH --cpus-per-task=1
#SBATCH --account=GEOG024542

module  load apps/singularity/1.1.3 lib/openmpi/4.0.2-gcc.4.8.5 

srun singularity exec \
-B/user/work/$(whoami):/user/work/$(whoami) \
/user/work/$(whoami)/Abil/singularity/abil.sif \
python /user/work/$(whoami)/Abil/cluster_example/hpc_post.py 

export SINGULARITY_CACHEDIR=/user/work/$(whoami)/.singularity
