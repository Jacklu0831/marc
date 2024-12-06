# python make_sbatch.py --time 48 --bash_files bash_commands/1205_llama1b_prefix_tuning/1205_0_llama1b_ttt_prefix_tuning_train.sh

# ttt prefix tuning with llama1b lr1e-2 train
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1205_0_llama1b_ttt_prefix_tuning_train_lr1e-2 \
    --data_file kaggle_dataset/arc-agi_training_challenges.json \
    --batch_size 2 \
    --epochs 2 \
    --num_virtual_tokens 25 \
    --learning_rate 1e-2 \
    --new_format \
    --flash_attn

# ttt prefix tuning with llama1b lr5e-3 train
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1205_0_llama1b_ttt_prefix_tuning_train_lr5e-3 \
    --data_file kaggle_dataset/arc-agi_training_challenges.json \
    --batch_size 2 \
    --epochs 2 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --flash_attn

# ttt prefix tuning with llama1b lr1e-3 train
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1205_0_llama1b_ttt_prefix_tuning_train_lr1e-3 \
    --data_file kaggle_dataset/arc-agi_training_challenges.json \
    --batch_size 2 \
    --epochs 2 \
    --num_virtual_tokens 25 \
    --learning_rate 1e-3 \
    --new_format \
    --flash_attn

# Submitted batch job 54381320
# Submitted batch job 54381321
# Submitted batch job 54381322