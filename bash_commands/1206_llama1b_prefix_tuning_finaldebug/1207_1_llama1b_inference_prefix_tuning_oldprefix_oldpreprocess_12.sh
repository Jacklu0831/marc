# python make_sbatch.py --time 48 --bash_files bash_commands/1206_llama1b_prefix_tuning_finaldebug/1207_1_llama1b_inference_prefix_tuning_oldprefix_oldpreprocess_12.sh

# llama1b epoch2 oldttt oldpreprocess seed0
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1207_1_llama1b_inference_prefix_tuning_oldprefix_oldpreprocess_seed0 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed0 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# llama1b epoch2 oldttt oldpreprocess seed1
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1207_1_llama1b_inference_prefix_tuning_oldprefix_oldpreprocess_seed1 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed1 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# llama1b epoch2 oldttt oldpreprocess seed2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1207_1_llama1b_inference_prefix_tuning_oldprefix_oldpreprocess_seed2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed2 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# llama1b epoch2 oldttt oldpreprocess seed3
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1207_1_llama1b_inference_prefix_tuning_oldprefix_oldpreprocess_seed3 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed3 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# llama1b epoch2 oldttt oldpreprocess seed4
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1207_1_llama1b_inference_prefix_tuning_oldprefix_oldpreprocess_seed4 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed4 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# llama1b epoch2 oldttt oldpreprocess seed5
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1207_1_llama1b_inference_prefix_tuning_oldprefix_oldpreprocess_seed5 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed5 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# llama1b epoch2 oldttt oldpreprocess seed6
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1207_1_llama1b_inference_prefix_tuning_oldprefix_oldpreprocess_seed6 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed6 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# llama1b epoch2 oldttt oldpreprocess seed7
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1207_1_llama1b_inference_prefix_tuning_oldprefix_oldpreprocess_seed7 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed7 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# llama1b epoch2 oldttt oldpreprocess seed8
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1207_1_llama1b_inference_prefix_tuning_oldprefix_oldpreprocess_seed8 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed8 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# llama1b epoch2 oldttt oldpreprocess seed9
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1207_1_llama1b_inference_prefix_tuning_oldprefix_oldpreprocess_seed9 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1207_0_llama1b_ttt_prefix_tuning_oldprefix_oldpreprocess_seed9 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# Submitted batch job 54531731
# Submitted batch job 54531732
# Submitted batch job 54531733
# Submitted batch job 54531734
# Submitted batch job 54531735
# Submitted batch job 54531736
# Submitted batch job 54531737
# Submitted batch job 54531738
# Submitted batch job 54531739
# Submitted batch job 54531740
