# python make_sbatch.py --time 48 --bash_files bash_commands/1210_llama1b_prefix_tuning_ntoken_lr_epoch_gridsearch/1210_0_ttt_search_lr3e-1_epoch10.sh

# ttt search lr3e-1 epoch10 ntoken1
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1210_0_ttt_search_lr3e-1_epoch10_ntoken1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 1 \
    --learning_rate 3e-1 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb

# ttt search lr3e-1 epoch10 ntoken5
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1210_0_ttt_search_lr3e-1_epoch10_ntoken5 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 5 \
    --learning_rate 3e-1 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb

# ttt search lr3e-1 epoch10 ntoken10
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1210_0_ttt_search_lr3e-1_epoch10_ntoken10 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 10 \
    --learning_rate 3e-1 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb

# ttt search lr3e-1 epoch10 ntoken50
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1210_0_ttt_search_lr3e-1_epoch10_ntoken50 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 50 \
    --learning_rate 3e-1 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb

# done
# Submitted batch job 54744081
# Submitted batch job 54744082
# Submitted batch job 54744083
# Submitted batch job 54744084