# python make_sbatch.py --time 48 --bash_files bash_commands/1205_llama1b_prefix_tuning/1205_1_llama1b_inference_prefix_tuning_eval_noflashattn.sh

# inference prefix tuning with llama1b lr1e-2 epoch1
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1205_1_llama1b_inference_prefix_tuning_eval_lr1e-2_noflashattn_epoch1 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1205_0_llama1b_ttt_prefix_tuning_eval_lr1e-2 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# inference prefix tuning with llama1b lr5e-3 epoch1
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1205_1_llama1b_inference_prefix_tuning_eval_lr5e-3_noflashattn_epoch1 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1205_0_llama1b_ttt_prefix_tuning_eval_lr5e-3 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# inference prefix tuning with llama1b lr1e-3 epoch1
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1205_1_llama1b_inference_prefix_tuning_eval_lr1e-3_noflashattn_epoch1 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1205_0_llama1b_ttt_prefix_tuning_eval_lr1e-3 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# inference prefix tuning with llama1b lr1e-2 epoch2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1205_1_llama1b_inference_prefix_tuning_eval_lr1e-2_noflashattn_epoch2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1205_0_llama1b_ttt_prefix_tuning_eval_lr1e-2 \
    --num_virtual_tokens 25 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# inference prefix tuning with llama1b lr5e-3 epoch2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1205_1_llama1b_inference_prefix_tuning_eval_lr5e-3_noflashattn_epoch2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1205_0_llama1b_ttt_prefix_tuning_eval_lr5e-3 \
    --num_virtual_tokens 25 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# inference prefix tuning with llama1b lr1e-3 epoch2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1205_1_llama1b_inference_prefix_tuning_eval_lr1e-3_noflashattn_epoch2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1205_0_llama1b_ttt_prefix_tuning_eval_lr1e-3 \
    --num_virtual_tokens 25 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# Submitted batch job 54403303
# Submitted batch job 54403304
# Submitted batch job 54403305
# Submitted batch job 54403306
# Submitted batch job 54403307
# Submitted batch job 54403308

# worse than with flashattn, wtf