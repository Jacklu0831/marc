# python make_sbatch.py --time 48 --bash_files bash_commands/1206_llama1b_prefix_tuning_finaldebug/1210_1_llama1b_inference_prefix_tuning_reuse_prefix_run2_epoch10.sh

# inference noreuseprefix_seed0_run2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1209_1_llama1b_inference_prefix_tuning_noreuseprefix_seed0_run2_epoch10 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1209_0_llama1b_ttt_prefix_tuning_noreuseprefix_seed0_run2_epoch10 \
    --num_virtual_tokens 25 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# inference noreuseprefix_seed1_run2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1209_1_llama1b_inference_prefix_tuning_noreuseprefix_seed1_run2_epoch10 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1209_0_llama1b_ttt_prefix_tuning_noreuseprefix_seed1_run2_epoch10 \
    --num_virtual_tokens 25 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# inference noreuseprefix_seed2_run2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1209_1_llama1b_inference_prefix_tuning_noreuseprefix_seed2_run2_epoch10 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1209_0_llama1b_ttt_prefix_tuning_noreuseprefix_seed2_run2_epoch10 \
    --num_virtual_tokens 25 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# inference reuseprefix_seed0_run2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1209_1_llama1b_inference_prefix_tuning_reuseprefix_seed0_run2_epoch10 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1209_0_llama1b_ttt_prefix_tuning_reuseprefix_seed0_run2_epoch10 \
    --num_virtual_tokens 25 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# inference reuseprefix_seed1_run2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1209_1_llama1b_inference_prefix_tuning_reuseprefix_seed1_run2_epoch10 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1209_0_llama1b_ttt_prefix_tuning_reuseprefix_seed1_run2_epoch10 \
    --num_virtual_tokens 25 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# inference reuseprefix_seed2_run2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1209_1_llama1b_inference_prefix_tuning_reuseprefix_seed2_run2_epoch10 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1209_0_llama1b_ttt_prefix_tuning_reuseprefix_seed2_run2_epoch10 \
    --num_virtual_tokens 25 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# reuse are terrible, noreuse are 5/6/10
# Submitted batch job 54643229
# Submitted batch job 54643230
# Submitted batch job 54643231