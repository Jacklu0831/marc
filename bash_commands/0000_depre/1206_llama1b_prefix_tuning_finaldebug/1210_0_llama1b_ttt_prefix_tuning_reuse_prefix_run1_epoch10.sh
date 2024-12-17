# python make_sbatch.py --time 48 --bash_files bash_commands/1206_llama1b_prefix_tuning_finaldebug/1210_0_llama1b_ttt_prefix_tuning_reuse_prefix_run1_epoch10.sh

# noreuse prefix seed 0 run1
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1209_0_llama1b_ttt_prefix_tuning_noreuseprefix_seed0_run1_epoch10 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --flash_attn \
    --logging_steps 1 \
    --seed 0

# noreuse prefix seed 1 run1
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1209_0_llama1b_ttt_prefix_tuning_noreuseprefix_seed1_run1_epoch10 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --flash_attn \
    --logging_steps 1 \
    --seed 1

# noreuse prefix seed 2 run1
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1209_0_llama1b_ttt_prefix_tuning_noreuseprefix_seed2_run1_epoch10 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --flash_attn \
    --logging_steps 1 \
    --seed 2

# reuse prefix seed 0 run1
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1209_0_llama1b_ttt_prefix_tuning_reuseprefix_seed0_run1_epoch10 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --flash_attn \
    --logging_steps 1 \
    --reuse_prefix \
    --seed 0

# reuse prefix seed 1 run1
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1209_0_llama1b_ttt_prefix_tuning_reuseprefix_seed1_run1_epoch10 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --flash_attn \
    --logging_steps 1 \
    --reuse_prefix \
    --seed 1

# reuse prefix seed 2 run1
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1209_0_llama1b_ttt_prefix_tuning_reuseprefix_seed2_run1_epoch10 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --flash_attn \
    --logging_steps 1 \
    --reuse_prefix \
    --seed 2

# Submitted batch job 54647719
# Submitted batch job 54647720
# Submitted batch job 54647721