# python make_sbatch.py --time 48 --bash_files bash_commands/1121_1_llama1b_ttt_pt_lr5e-3.sh
# train ttt with raw meta llama1b, can ablate lora_to_output

# ttt with llama1b ntoken5
python test_time_train_prompt_tuning.py \
    --pt_config configs/ttt/1B_prompt_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1119_1_llama1b_ttt_pt_lr5e-3_ntoken5 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 5 \
    --learning_rate 5e-3 \
    --new_format

# ttt with llama1b ntoken10
python test_time_train_prompt_tuning.py \
    --pt_config configs/ttt/1B_prompt_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1119_1_llama1b_ttt_pt_lr5e-3_ntoken10 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 10 \
    --learning_rate 5e-3 \
    --new_format

# ttt with llama1b ntoken25
python test_time_train_prompt_tuning.py \
    --pt_config configs/ttt/1B_prompt_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1119_1_llama1b_ttt_pt_lr5e-3_ntoken25 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format

# ttt with llama1b ntoken50
python test_time_train_prompt_tuning.py \
    --pt_config configs/ttt/1B_prompt_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1119_1_llama1b_ttt_pt_lr5e-3_ntoken50 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 50 \
    --learning_rate 5e-3 \
    --new_format

# Submitted batch job 53852290
# Submitted batch job 53852291
# Submitted batch job 53852292
# Submitted batch job 53852293