# python make_sbatch.py --time 48 --bash_files bash_commands/1126_2_llama1b_inference_prefix_tuning_ntoken1_lr1e-3.sh

# ntoken1 epoch1
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1126_2_llama1b_inference_prefix_tuning_ntoken1_lr1e-3_epoch1 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1126_0_llama1b_ttt_prefix_tuning_lr1e-3_ntoken1 \
    --num_virtual_tokens 1 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# ntoken1 epoch3
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1126_2_llama1b_inference_prefix_tuning_ntoken1_lr1e-3_epoch3 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1126_0_llama1b_ttt_prefix_tuning_lr1e-3_ntoken1 \
    --num_virtual_tokens 1 \
    --pt_epoch 3 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# ntoken1 epoch5
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1126_2_llama1b_inference_prefix_tuning_ntoken1_lr1e-3_epoch5 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1126_0_llama1b_ttt_prefix_tuning_lr1e-3_ntoken1 \
    --num_virtual_tokens 1 \
    --pt_epoch 5 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# ntoken1 epoch7
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1126_2_llama1b_inference_prefix_tuning_ntoken1_lr1e-3_epoch7 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1126_0_llama1b_ttt_prefix_tuning_lr1e-3_ntoken1 \
    --num_virtual_tokens 1 \
    --pt_epoch 7 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# ntoken1 epoch9
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1126_2_llama1b_inference_prefix_tuning_ntoken1_lr1e-3_epoch9 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1126_0_llama1b_ttt_prefix_tuning_lr1e-3_ntoken1 \
    --num_virtual_tokens 1 \
    --pt_epoch 9 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format
