# python make_sbatch.py --time 48 --bash_files bash_commands/1213_llama1b_prefix_tuning_metalora/1213_1_inference_lr1e-1_epoch20_ntoken1_train.sh

# inference lr1e-1 epoch20 ntoken1 train
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1213_1_inference_lr1e-1_epoch20_ntoken1_train \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1213_0_ttt_lr1e-1_epoch20_ntoken1_train \
    --num_virtual_tokens 1 \
    --pt_epoch 20 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_training_challenges.json \
    --solution_file kaggle_dataset/arc-agi_training_solutions.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# Submitted batch job 54821668