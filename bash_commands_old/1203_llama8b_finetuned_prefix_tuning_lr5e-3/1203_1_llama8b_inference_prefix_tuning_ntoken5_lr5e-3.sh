# python make_sbatch.py --time 48 --bash_files bash_commands/1203_llama8b_finetuned_prefix_tuning_lr5e-3/1203_1_llama8b_inference_prefix_tuning_ntoken5_lr5e-3.sh

# ntoken5 epoch2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1203_1_llama8b_inference_prefix_tuning_ntoken5_lr5e-3_epoch2 \
    --pretrained_checkpoint downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --pt_checkpoints_folder train_outputs/1203_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# ntoken5 epoch4
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1203_1_llama8b_inference_prefix_tuning_ntoken5_lr5e-3_epoch4 \
    --pretrained_checkpoint downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --pt_checkpoints_folder train_outputs/1203_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 4 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# ntoken5 epoch6
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1203_1_llama8b_inference_prefix_tuning_ntoken5_lr5e-3_epoch6 \
    --pretrained_checkpoint downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --pt_checkpoints_folder train_outputs/1203_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 6 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# ntoken5 epoch8
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1203_1_llama8b_inference_prefix_tuning_ntoken5_lr5e-3_epoch8 \
    --pretrained_checkpoint downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --pt_checkpoints_folder train_outputs/1203_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 8 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# ntoken5 epoch10
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1203_1_llama8b_inference_prefix_tuning_ntoken5_lr5e-3_epoch10 \
    --pretrained_checkpoint downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --pt_checkpoints_folder train_outputs/1203_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_ntoken5 \
    --num_virtual_tokens 5 \
    --pt_epoch 10 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# Submitted batch job 54408917
# Submitted batch job 54408918
# Submitted batch job 54408919
# Submitted batch job 54408920
# Submitted batch job 54408921