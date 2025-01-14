# python make_sbatch.py --time 48 --bash_files bash_commands/1216_llama8b_debug/1216_0_ttt_lr1e-1_epoch20_ntoken1_rerunsplit.sh

# ttt search lr1e-1 epoch20 ntoken1 split0
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1216_0_ttt_lr1e-1_epoch20_ntoken1_rerunsplit \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 20 \
    --num_virtual_tokens 1 \
    --learning_rate 1e-1 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb \
    --div 2 \
    --mod 0

# ttt search lr1e-1 epoch20 ntoken1 split1
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1216_0_ttt_lr1e-1_epoch20_ntoken1_rerunsplit \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 20 \
    --num_virtual_tokens 1 \
    --learning_rate 1e-1 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb \
    --div 2 \
    --mod 1

# Submitted batch job 55012267
# Submitted batch job 55012268

# rerun
# Submitted batch job 55062661
# Submitted batch job 55062662