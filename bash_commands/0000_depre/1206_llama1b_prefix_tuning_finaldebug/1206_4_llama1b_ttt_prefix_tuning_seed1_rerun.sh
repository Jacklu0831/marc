# python make_sbatch.py --time 48 --bash_files bash_commands/1206_llama1b_prefix_tuning_finaldebug/1206_4_llama1b_ttt_prefix_tuning_seed1_rerun.sh

# ttt prefix tuning with llama1b ntoken25 old
python test_time_train_prefix_tuning_old.py \
    --pt_config configs/ttt/1B_prefix_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1206_4_llama1b_ttt_prefix_tuning_old_seed1_rerun \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 2 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --seed 1

# ttt prefix tuning with llama1b ntoken25 new
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1206_4_llama1b_ttt_prefix_tuning_new_seed1_rerun \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 2 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --seed 1

# ttt prefix tuning with llama1b ntoken25 new flashattn
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1206_4_llama1b_ttt_prefix_tuning_new_flashattn_seed1_rerun \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 2 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --seed 1 \
    --flash_attn

# Submitted batch job 54474356
# Submitted batch job 54474357
# Submitted batch job 54474358