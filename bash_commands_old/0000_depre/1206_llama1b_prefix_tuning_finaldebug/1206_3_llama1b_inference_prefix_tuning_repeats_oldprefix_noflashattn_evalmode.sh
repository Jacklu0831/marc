# python make_sbatch.py --time 48 --bash_files bash_commands/1206_llama1b_prefix_tuning_finaldebug/1206_3_llama1b_inference_prefix_tuning_repeats_oldprefix_noflashattn_evalmode.sh

# llama1b oldprefix noflashattn evalmode epoch2 seed1
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1206_3_llama1b_inference_prefix_tuning_repeats_oldprefix_noflashattn_evalmode_epoch2_seed1 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1126_0_llama1b_ttt_prefix_tuning_lr5e-3_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --seed 1

# llama1b oldprefix noflashattn evalmode epoch2 seed2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1206_3_llama1b_inference_prefix_tuning_repeats_oldprefix_noflashattn_evalmode_epoch2_seed2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1126_0_llama1b_ttt_prefix_tuning_lr5e-3_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --seed 2

# llama1b oldprefix noflashattn evalmode epoch2 seed3
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1206_3_llama1b_inference_prefix_tuning_repeats_oldprefix_noflashattn_evalmode_epoch2_seed3 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1126_0_llama1b_ttt_prefix_tuning_lr5e-3_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --seed 3

# Submitted batch job 54417123
# Submitted batch job 54417124
# Submitted batch job 54417125