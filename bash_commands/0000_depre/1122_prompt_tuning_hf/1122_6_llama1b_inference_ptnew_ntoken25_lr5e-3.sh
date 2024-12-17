# python make_sbatch.py --time 48 --bash_files bash_commands/1122_6_llama1b_inference_ptnew_ntoken25_lr5e-3.sh

# ntoken25 epoch1
python predict_prompt_tuning.py \
    --experiment_folder inference_outputs/experiments/1122_6_llama1b_inference_ptnew_ntoken25_lr5e-3_epoch1 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1119_1_llama1b_ttt_pt_lr5e-3_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# ntoken25 epoch3
python predict_prompt_tuning.py \
    --experiment_folder inference_outputs/experiments/1122_6_llama1b_inference_ptnew_ntoken25_lr5e-3_epoch3 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1119_1_llama1b_ttt_pt_lr5e-3_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 3 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# ntoken25 epoch5
python predict_prompt_tuning.py \
    --experiment_folder inference_outputs/experiments/1122_6_llama1b_inference_ptnew_ntoken25_lr5e-3_epoch5 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1119_1_llama1b_ttt_pt_lr5e-3_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 5 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# ntoken25 epoch7
python predict_prompt_tuning.py \
    --experiment_folder inference_outputs/experiments/1122_6_llama1b_inference_ptnew_ntoken25_lr5e-3_epoch7 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1119_1_llama1b_ttt_pt_lr5e-3_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 7 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# ntoken25 epoch9
python predict_prompt_tuning.py \
    --experiment_folder inference_outputs/experiments/1122_6_llama1b_inference_ptnew_ntoken25_lr5e-3_epoch9 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1119_1_llama1b_ttt_pt_lr5e-3_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 9 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# Submitted batch job 53901048
# Submitted batch job 53901049
# Submitted batch job 53901050
# Submitted batch job 53901051
# Submitted batch job 53901052
# OOM