# python make_sbatch.py --time 48 --bash_files bash_commands/1210_llama1b_prefix_tuning_ntoken_lr_epoch_gridsearch/1210_0_ttt_search_lr1e-2_epoch20.sh

# ttt search lr1e-2 epoch20 ntoken1
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1210_0_ttt_search_lr1e-2_epoch20_ntoken1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 20 \
    --num_virtual_tokens 1 \
    --learning_rate 1e-2 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb

# ttt search lr1e-2 epoch20 ntoken5
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1210_0_ttt_search_lr1e-2_epoch20_ntoken5 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 20 \
    --num_virtual_tokens 5 \
    --learning_rate 1e-2 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb

# ttt search lr1e-2 epoch20 ntoken10
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1210_0_ttt_search_lr1e-2_epoch20_ntoken10 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 20 \
    --num_virtual_tokens 10 \
    --learning_rate 1e-2 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb

# ttt search lr1e-2 epoch20 ntoken50
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1210_0_ttt_search_lr1e-2_epoch20_ntoken50 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 20 \
    --num_virtual_tokens 50 \
    --learning_rate 1e-2 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb

# done
# Submitted batch job 54649232
# Submitted batch job 54649233
# Submitted batch job 54649234
# Submitted batch job 54649235