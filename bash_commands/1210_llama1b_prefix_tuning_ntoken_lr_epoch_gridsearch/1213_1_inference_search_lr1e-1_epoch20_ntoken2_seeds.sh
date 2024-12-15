# python make_sbatch.py --time 48 --bash_files bash_commands/1210_llama1b_prefix_tuning_ntoken_lr_epoch_gridsearch/1213_1_inference_search_lr1e-1_epoch20_ntoken2_seeds.sh

# inference lr1e-1 epoch20 ntoken2 seed1
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1213_1_inference_search_lr1e-1_epoch20_ntoken2_seed1 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1213_0_ttt_search_lr1e-1_epoch20_ntoken2_seed1 \
    --num_virtual_tokens 2 \
    --pt_epoch 20 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --limit_tokens

# inference lr1e-1 epoch20 ntoken2 seed2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1213_1_inference_search_lr1e-1_epoch20_ntoken2_seed2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1213_0_ttt_search_lr1e-1_epoch20_ntoken2_seed2 \
    --num_virtual_tokens 2 \
    --pt_epoch 20 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --limit_tokens

# inference lr1e-1 epoch20 ntoken2 seed3
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1213_1_inference_search_lr1e-1_epoch20_ntoken2_seed3 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1213_0_ttt_search_lr1e-1_epoch20_ntoken2_seed3 \
    --num_virtual_tokens 2 \
    --pt_epoch 20 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --limit_tokens

# Submitted batch job 54841820
# Submitted batch job 54841821
# Submitted batch job 54841822