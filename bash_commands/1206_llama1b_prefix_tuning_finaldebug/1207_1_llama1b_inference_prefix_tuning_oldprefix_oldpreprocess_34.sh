# python make_sbatch.py --time 48 --bash_files bash_commands/1206_llama1b_prefix_tuning_finaldebug/1207_1_llama1b_inference_prefix_tuning_oldprefix_oldpreprocess_34.sh

# llama1b epoch2 oldttt oldpreprocess seed10
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1207_1_llama1b_inference_prefix_tuning_oldprefix_oldpreprocess_seed10 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed10 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# llama1b epoch2 oldttt oldpreprocess seed11
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1207_1_llama1b_inference_prefix_tuning_oldprefix_oldpreprocess_seed11 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed11 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# llama1b epoch2 oldttt oldpreprocess seed12
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1207_1_llama1b_inference_prefix_tuning_oldprefix_oldpreprocess_seed12 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed12 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# llama1b epoch2 oldttt oldpreprocess seed13
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1207_1_llama1b_inference_prefix_tuning_oldprefix_oldpreprocess_seed13 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed13 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# llama1b epoch2 oldttt oldpreprocess seed14
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1207_1_llama1b_inference_prefix_tuning_oldprefix_oldpreprocess_seed14 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed14 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# llama1b epoch2 oldttt oldpreprocess seed15
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1207_1_llama1b_inference_prefix_tuning_oldprefix_oldpreprocess_seed15 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed15 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# llama1b epoch2 oldttt oldpreprocess seed16
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1207_1_llama1b_inference_prefix_tuning_oldprefix_oldpreprocess_seed16 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed16 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# llama1b epoch2 oldttt oldpreprocess seed17
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1207_1_llama1b_inference_prefix_tuning_oldprefix_oldpreprocess_seed17 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed17 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# llama1b epoch2 oldttt oldpreprocess seed18
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1207_1_llama1b_inference_prefix_tuning_oldprefix_oldpreprocess_seed18 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed18 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# llama1b epoch2 oldttt oldpreprocess seed19
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1207_1_llama1b_inference_prefix_tuning_oldprefix_oldpreprocess_seed19 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed19 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# Submitted batch job 54534227
# Submitted batch job 54534228
# Submitted batch job 54534229
# Submitted batch job 54534230
# Submitted batch job 54534231
# Submitted batch job 54534232
# Submitted batch job 54534233
# Submitted batch job 54534234
# Submitted batch job 54534235
# Submitted batch job 54534236