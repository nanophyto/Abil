#!/bin/bash
#
#
#SBATCH --time=0-6:00:00
#SBATCH --nodes=1
#SBATCH --mem=10000M
#SBATCH --cpus-per-task=16
#SBATCH --array=0-1

i=${SLURM_ARRAY_TASK_ID}

module  load apptainer/1.3.1

singularity exec \
-B/user/work/$(whoami):/user/work/$(whoami) \
/user/work/$(whoami)/Abil/singularity/abil.sif \
python /user/work/$(whoami)/Abil/hpc_tune.py ${SLURM_CPUS_PER_TASK} ${i} "xgb"

export SINGULARITY_CACHEDIR=/user/work/$(whoami)/.singularity
