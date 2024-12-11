# python make_sbatch.py --time 48 --bash_files bash_commands/1126_llama1b_prefix_tuning_lr5e-3/1126_2_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_rerun.sh

# ntoken25 epoch1 rerun
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1126_2_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_rerun_epoch1 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1126_0_llama1b_ttt_prefix_tuning_lr5e-3_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# ntoken25 epoch3 rerun
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1126_2_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_rerun_epoch3 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1126_0_llama1b_ttt_prefix_tuning_lr5e-3_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 3 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# ntoken25 epoch5 rerun
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1126_2_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_rerun_epoch5 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1126_0_llama1b_ttt_prefix_tuning_lr5e-3_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 5 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# ntoken25 epoch7 rerun
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1126_2_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_rerun_epoch7 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1126_0_llama1b_ttt_prefix_tuning_lr5e-3_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 7 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# ntoken25 epoch9 rerun
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1126_2_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_rerun_epoch9 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1126_0_llama1b_ttt_prefix_tuning_lr5e-3_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 9 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# Submitted batch job 54129361
# Submitted batch job 54129362
# Submitted batch job 54129363
# Submitted batch job 54129364
# Submitted batch job 54129365

# 7, 7, 9, 8, 9