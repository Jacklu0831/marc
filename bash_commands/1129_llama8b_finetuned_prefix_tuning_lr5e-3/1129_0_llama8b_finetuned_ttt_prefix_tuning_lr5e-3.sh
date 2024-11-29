# python make_sbatch.py --time 48 --bash_files bash_commands/1129_llama8b_finetuned_prefix_tuning_lr5e-3/1129_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3.sh

# ttt prefix tuning with llama1b ntoken1
python test_time_train_prefix_tuning.py \
    --pt_config configs/ttt/8B_finetuned_prefix_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1129_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_ntoken1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 1 \
    --learning_rate 5e-3 \
    --new_format

# ttt prefix tuning with llama1b ntoken3
python test_time_train_prefix_tuning.py \
    --pt_config configs/ttt/8B_finetuned_prefix_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1129_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_ntoken3 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 3 \
    --learning_rate 5e-3 \
    --new_format

# ttt prefix tuning with llama1b ntoken5
python test_time_train_prefix_tuning.py \
    --pt_config configs/ttt/8B_finetuned_prefix_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1129_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_ntoken5 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 5 \
    --learning_rate 5e-3 \
    --new_format

# ttt prefix tuning with llama1b ntoken10
python test_time_train_prefix_tuning.py \
    --pt_config configs/ttt/8B_finetuned_prefix_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1129_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_ntoken10 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 10 \
    --learning_rate 5e-3 \
    --new_format

# ttt prefix tuning with llama1b ntoken25
python test_time_train_prefix_tuning.py \
    --pt_config configs/ttt/8B_finetuned_prefix_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1129_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_ntoken25 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format

# ttt prefix tuning with llama1b ntoken50
python test_time_train_prefix_tuning.py \
    --pt_config configs/ttt/8B_finetuned_prefix_tuning_single_device.yaml \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1129_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_ntoken50 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 50 \
    --learning_rate 5e-3 \
    --new_format

# Submitted batch job 54111433
# Submitted batch job 54111434
# Submitted batch job 54111435
# Submitted batch job 54111436
# Submitted batch job 54111437
# Submitted batch job 54111438