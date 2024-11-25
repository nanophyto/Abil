#!/bin/bash
#
#
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=100000M
#SBATCH --cpus-per-task=16
#SBATCH --array=0-1
#SBATCH --account=GEOG024542

i=${SLURM_ARRAY_TASK_ID}

module  load apptainer/1.1.9 openmpi/4.1.2 

srun singularity exec \
-B/user/work/$(whoami):/user/work/$(whoami) \
/user/work/$(whoami)/Abil/singularity/abil.sif \
python /user/work/$(whoami)/Abil/wiseman2024/studies/hpc_predict.py ${SLURM_CPUS_PER_TASK} ${i}

export SINGULARITY_CACHEDIR=/user/work/$(whoami)/.singularity
