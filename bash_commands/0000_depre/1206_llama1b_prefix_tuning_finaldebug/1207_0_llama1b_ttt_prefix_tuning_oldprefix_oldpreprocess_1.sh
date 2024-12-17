# python make_sbatch.py --time 48 --bash_files bash_commands/1206_llama1b_prefix_tuning_finaldebug/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_1.sh

# ttt prefix tuning oldprefix seed0
python test_time_train_prefix_tuning_old.py \
    --pt_config configs/ttt/1B_prefix_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed0 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 2 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --use_old_preprocess \
    --seed 0

# ttt prefix tuning oldprefix seed1
python test_time_train_prefix_tuning_old.py \
    --pt_config configs/ttt/1B_prefix_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 2 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --use_old_preprocess \
    --seed 1

# ttt prefix tuning oldprefix seed2
python test_time_train_prefix_tuning_old.py \
    --pt_config configs/ttt/1B_prefix_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed2 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 2 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --use_old_preprocess \
    --seed 2

# ttt prefix tuning oldprefix seed3
python test_time_train_prefix_tuning_old.py \
    --pt_config configs/ttt/1B_prefix_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed3 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 2 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --use_old_preprocess \
    --seed 3

# ttt prefix tuning oldprefix seed4
python test_time_train_prefix_tuning_old.py \
    --pt_config configs/ttt/1B_prefix_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed4 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 2 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --use_old_preprocess \
    --seed 4

# Submitted batch job 54497851
# Submitted batch job 54497852
# Submitted batch job 54497853
# Submitted batch job 54497854
# Submitted batch job 54497855