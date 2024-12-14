# python make_sbatch.py --time 48 --bash_files bash_commands/1210_llama1b_prefix_tuning_ntoken_lr_epoch_gridsearch/1210_1_inference_search_lr3e-3_epoch2.sh

# inference search lr3e-3 epoch2 ntoken1
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1210_1_inference_search_lr3e-3_epoch2_ntoken1 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1210_0_ttt_search_lr3e-3_epoch2_ntoken1 \
    --num_virtual_tokens 1 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --limit_tokens

# inference search lr3e-3 epoch2 ntoken5
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1210_1_inference_search_lr3e-3_epoch2_ntoken5 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1210_0_ttt_search_lr3e-3_epoch2_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --limit_tokens

# inference search lr3e-3 epoch2 ntoken10
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1210_1_inference_search_lr3e-3_epoch2_ntoken10 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1210_0_ttt_search_lr3e-3_epoch2_ntoken10 \
    --num_virtual_tokens 10 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --limit_tokens

# inference search lr3e-3 epoch2 ntoken50
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1210_1_inference_search_lr3e-3_epoch2_ntoken50 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1210_0_ttt_search_lr3e-3_epoch2_ntoken50 \
    --num_virtual_tokens 50 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --limit_tokens

# Submitted batch job 54672803
# Submitted batch job 54672804
# Submitted batch job 54672805
# Submitted batch job 54672806