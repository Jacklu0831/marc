# python make_sbatch.py --time 48 --bash_files bash_commands/1125_0_llama1b_ttt_pt_lr5e-3_few_token.sh
# train ttt with raw meta llama1b, can ablate lora_to_output

# ttt with llama1b ntoken1
python test_time_train_prompt_tuning.py \
    --pt_config configs/ttt/1B_prompt_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1119_1_llama1b_ttt_pt_lr5e-3_ntoken1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 1 \
    --learning_rate 5e-3 \
    --new_format

# ttt with llama1b ntoken2
python test_time_train_prompt_tuning.py \
    --pt_config configs/ttt/1B_prompt_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1119_1_llama1b_ttt_pt_lr5e-3_ntoken2 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 2 \
    --learning_rate 5e-3 \
    --new_format

# ttt with llama1b ntoken3
python test_time_train_prompt_tuning.py \
    --pt_config configs/ttt/1B_prompt_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1119_1_llama1b_ttt_pt_lr5e-3_ntoken3 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 3 \
    --learning_rate 5e-3 \
    --new_format

# ttt with llama1b ntoken4
python test_time_train_prompt_tuning.py \
    --pt_config configs/ttt/1B_prompt_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1119_1_llama1b_ttt_pt_lr5e-3_ntoken4 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 4 \
    --learning_rate 5e-3 \
    --new_format

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

# Submitted batch job 53999123
# Submitted batch job 53999124
# Submitted batch job 53999125
# Submitted batch job 53999126
# Submitted batch job 54000110