# python make_sbatch.py --time 48 --bash_files bash_commands/1206_llama1b_prefix_tuning_finaldebug/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_2.sh

# ttt prefix tuning oldprefix seed5
python test_time_train_prefix_tuning_old.py \
    --pt_config configs/ttt/1B_prefix_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed5 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 2 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --use_old_preprocess \
    --seed 5

# ttt prefix tuning oldprefix seed6
python test_time_train_prefix_tuning_old.py \
    --pt_config configs/ttt/1B_prefix_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed6 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 2 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --use_old_preprocess \
    --seed 6

# ttt prefix tuning oldprefix seed7
python test_time_train_prefix_tuning_old.py \
    --pt_config configs/ttt/1B_prefix_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed7 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 2 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --use_old_preprocess \
    --seed 7

# ttt prefix tuning oldprefix seed8
python test_time_train_prefix_tuning_old.py \
    --pt_config configs/ttt/1B_prefix_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed8 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 2 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --use_old_preprocess \
    --seed 8

# ttt prefix tuning oldprefix seed9
python test_time_train_prefix_tuning_old.py \
    --pt_config configs/ttt/1B_prefix_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed9 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 2 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --use_old_preprocess \
    --seed 9

# Submitted batch job 54497856
# Submitted batch job 54497857
# Submitted batch job 54497858
# Submitted batch job 54497859
# Submitted batch job 54497860