# python make_sbatch.py --time 48 --bash_files bash_commands/1206_llama1b_prefix_tuning_finaldebug/1206_5_llama1b_inference_prefix_tuning_seed1_rerun.sh

# llama1b epoch2 seed1 oldttt
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1206_5_llama1b_inference_prefix_tuning_oldttt_epoch2_seed1_rerun \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1206_4_llama1b_ttt_prefix_tuning_old_seed1_rerun \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --seed 1

# llama1b epoch2 seed1 newttt
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1206_5_llama1b_inference_prefix_tuning_newttt_epoch2_seed1_rerun \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1206_4_llama1b_ttt_prefix_tuning_new_seed1_rerun \
    --num_virtual_tokens 25 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --seed 1

# llama1b epoch2 seed1 newtttflash
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1206_5_llama1b_inference_prefix_tuning_newtttflash_epoch2_seed1_rerun \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1206_4_llama1b_ttt_prefix_tuning_new_flashattn_seed1_rerun \
    --num_virtual_tokens 25 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --seed 1

# Submitted batch job 54496423
# Submitted batch job 54496424
# Submitted batch job 54496425