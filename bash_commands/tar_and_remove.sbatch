#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=tar_and_remove
#SBATCH --output=slurm_outs/%j.out

module purge

singularity exec --nv \\
    --overlay /scratch/yl11330/my_env/overlay-50G-10M-pytorch.ext3:ro \\
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \\
    /bin/bash -c "source /ext3/env.sh; cd /scratch/yl11330/marc; conda activate ./penv; \\
        python tar_and_remove.py --date_end 1129"

# Submitted batch job 55012161