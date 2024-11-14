#!/bin/bash

#SBATCH --constraint='a100|h100'
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=ttt_with_llama1b_0

module purge

singularity exec --nv \
    --overlay /scratch/yl11330/my_env/overlay-50G-10M-pytorch.ext3:ro \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "source /ext3/env.sh; cd /scratch/yl11330/marc; conda activate ./penv; \
        python test_time_train.py --lora_config configs/ttt/1B_lora_single_device.yaml --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct --experiment_folder train_outputs/1114_llama1b_ttt --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json --batch_size 2 --epochs 2 --lora_rank 128 --lora_alpha 16.0 --learning_rate 5e-5 --new_format"
