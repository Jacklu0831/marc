# python make_sbatch.py --time 48 --bash_files bash_commands/1126_0_llama1b_ttt_prefix_tuning_lr5e-3.sh
# train ttt with raw meta llama1b, can ablate lora_to_output

# ttt prefix tuning with llama1b ntoken1
python test_time_train_prefix_tuning.py \
    --pt_config configs/ttt/1B_prefix_tuning_single_device.yaml \
    --base_checkpoint_dir meta-llama/Llama-3.1-8B-Instruct \
    --experiment_folder train_outputs/1126_0_llama1b_ttt_prefix_tuning_lr5e-3_ntoken1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 1 \
    --learning_rate 5e-3 \
    --new_format

# ttt prefix tuning with llama1b ntoken3
python test_time_train_prefix_tuning.py \
    --pt_config configs/ttt/1B_prefix_tuning_single_device.yaml \
    --base_checkpoint_dir meta-llama/Llama-3.1-8B-Instruct \
    --experiment_folder train_outputs/1126_0_llama1b_ttt_prefix_tuning_lr5e-3_ntoken3 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 3 \
    --learning_rate 5e-3 \
    --new_format

# ttt prefix tuning with llama1b ntoken5
python test_time_train_prefix_tuning.py \
    --pt_config configs/ttt/1B_prefix_tuning_single_device.yaml \
    --base_checkpoint_dir meta-llama/Llama-3.1-8B-Instruct \
    --experiment_folder train_outputs/1126_0_llama1b_ttt_prefix_tuning_lr5e-3_ntoken5 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 5 \
    --learning_rate 5e-3 \
    --new_format

# ttt prefix tuning with llama1b ntoken10
python test_time_train_prefix_tuning.py \
    --pt_config configs/ttt/1B_prefix_tuning_single_device.yaml \
    --base_checkpoint_dir meta-llama/Llama-3.1-8B-Instruct \
    --experiment_folder train_outputs/1126_0_llama1b_ttt_prefix_tuning_lr5e-3_ntoken10 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 10 \
    --learning_rate 5e-3 \
    --new_format

# ttt prefix tuning with llama1b ntoken25
python test_time_train_prefix_tuning.py \
    --pt_config configs/ttt/1B_prefix_tuning_single_device.yaml \
    --base_checkpoint_dir meta-llama/Llama-3.1-8B-Instruct \
    --experiment_folder train_outputs/1126_0_llama1b_ttt_prefix_tuning_lr5e-3_ntoken25 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format

# ttt prefix tuning with llama1b ntoken50
python test_time_train_prefix_tuning.py \
    --pt_config configs/ttt/1B_prefix_tuning_single_device.yaml \
    --base_checkpoint_dir meta-llama/Llama-3.1-8B-Instruct \
    --experiment_folder train_outputs/1126_0_llama1b_ttt_prefix_tuning_lr5e-3_ntoken50 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 50 \
    --learning_rate 5e-3 \
    --new_format

# Submitted batch job 54049799
# Submitted batch job 54049800
# Submitted batch job 54049801
# Submitted batch job 54049802
# Submitted batch job 54049803
# Submitted batch job 54049804