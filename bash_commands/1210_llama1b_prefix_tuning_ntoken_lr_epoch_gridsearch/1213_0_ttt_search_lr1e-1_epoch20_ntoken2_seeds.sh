# python make_sbatch.py --time 48 --bash_files bash_commands/1210_llama1b_prefix_tuning_ntoken_lr_epoch_gridsearch/1213_0_ttt_search_lr1e-1_epoch20_ntoken2_seeds.sh

# ttt search lr1e-1 epoch20 ntoken2 seed1
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1213_0_ttt_search_lr1e-1_epoch20_ntoken2_seed1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 20 \
    --num_virtual_tokens 2 \
    --learning_rate 1e-1 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb \
    --seed 1

# ttt search lr1e-1 epoch20 ntoken2 seed2
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1213_0_ttt_search_lr1e-1_epoch20_ntoken2_seed2 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 20 \
    --num_virtual_tokens 2 \
    --learning_rate 1e-1 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb \
    --seed 2

# ttt search lr1e-1 epoch20 ntoken2 seed3
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1213_0_ttt_search_lr1e-1_epoch20_ntoken2_seed3 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 20 \
    --num_virtual_tokens 2 \
    --learning_rate 1e-1 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb \
    --seed 3

# Submitted batch job 54798928
# Submitted batch job 54798929
# Submitted batch job 54798930