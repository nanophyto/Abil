#!/bin/bash
#
#
#SBATCH --time=0-6:00:00
#SBATCH --nodes=1
#SBATCH --mem=10000M
#SBATCH --cpus-per-task=16
#SBATCH --array=0-1
#SBATCH --account=GEOG024542

i=${SLURM_ARRAY_TASK_ID}

module  load apptainer/1.3.1 

singularity exec \
-B/user/work/$(whoami):/user/work/$(whoami) \
/user/work/$(whoami)/Abil/studies/wiseman2024/singularity/abil.sif \
python /user/work/$(whoami)/Abil/studies/wiseman2024/hpc_tune.py ${SLURM_CPUS_PER_TASK} ${i} "knn"

export SINGULARITY_CACHEDIR=/user/work/$(whoami)/.singularity
