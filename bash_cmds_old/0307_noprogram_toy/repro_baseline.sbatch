#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=6:00:00
#SBATCH --mem=12GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=toy_baseline
#SBATCH --output=slurm_outs/%j.out

module purge

MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

singularity exec --nv \
    --overlay /scratch/yl11330/my_env/overlay-50G-10M-pytorch.ext3:ro \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "source /ext3/env.sh; cd /scratch/yl11330/marc; conda activate ./penv; export MASTER_PORT; \
        python encoder_decoder_noprogram_toy/fit_2layernn.py"

# Submitted batch job 58079768
# Submitted batch job 58079769
# Submitted batch job 58079770