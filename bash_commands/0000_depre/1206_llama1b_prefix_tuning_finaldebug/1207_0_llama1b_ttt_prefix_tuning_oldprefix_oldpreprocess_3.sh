# python make_sbatch.py --time 48 --bash_files bash_commands/1206_llama1b_prefix_tuning_finaldebug/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_3.sh

# ttt prefix tuning oldprefix seed10
python test_time_train_prefix_tuning_old.py \
    --pt_config configs/ttt/1B_prefix_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed10 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 2 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --use_old_preprocess \
    --seed 10

# ttt prefix tuning oldprefix seed11
python test_time_train_prefix_tuning_old.py \
    --pt_config configs/ttt/1B_prefix_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed11 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 2 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --use_old_preprocess \
    --seed 11

# ttt prefix tuning oldprefix seed12
python test_time_train_prefix_tuning_old.py \
    --pt_config configs/ttt/1B_prefix_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed12 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 2 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --use_old_preprocess \
    --seed 12

# ttt prefix tuning oldprefix seed13
python test_time_train_prefix_tuning_old.py \
    --pt_config configs/ttt/1B_prefix_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed13 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 2 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --use_old_preprocess \
    --seed 13

# ttt prefix tuning oldprefix seed14
python test_time_train_prefix_tuning_old.py \
    --pt_config configs/ttt/1B_prefix_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed14 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 2 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --use_old_preprocess \
    --seed 14

# Submitted batch job 54529247
# Submitted batch job 54529248
# Submitted batch job 54529249
# Submitted batch job 54529250
# Submitted batch job 54529251