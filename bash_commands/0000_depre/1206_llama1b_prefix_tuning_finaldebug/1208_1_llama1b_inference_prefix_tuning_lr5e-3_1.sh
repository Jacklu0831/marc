# python make_sbatch.py --time 48 --bash_files bash_commands/1206_llama1b_prefix_tuning_finaldebug/1208_1_llama1b_inference_prefix_tuning_lr5e-3_1.sh

# ntoken1 epoch2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1208_1_llama1b_inference_prefix_tuning_ntoken1_lr5e-3_1_epoch2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1208_0_llama1b_ttt_prefix_tuning_lr5e-3_1_token1 \
    --num_virtual_tokens 1 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# ntoken3 epoch2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1208_1_llama1b_inference_prefix_tuning_ntoken3_lr5e-3_1_epoch2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1208_0_llama1b_ttt_prefix_tuning_lr5e-3_1_token3 \
    --num_virtual_tokens 3 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# ntoken5 epoch2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1208_1_llama1b_inference_prefix_tuning_ntoken5_lr5e-3_1_epoch2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1208_0_llama1b_ttt_prefix_tuning_lr5e-3_1_token3 \
    --num_virtual_tokens 5 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# ntoken10 epoch2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1208_1_llama1b_inference_prefix_tuning_ntoken10_lr5e-3_1_epoch2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1208_0_llama1b_ttt_prefix_tuning_lr5e-3_1_token3 \
    --num_virtual_tokens 10 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# ntoken25 epoch2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1208_1_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_1_epoch2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1208_0_llama1b_ttt_prefix_tuning_lr5e-3_1_token3 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# ntoken50 epoch2
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1208_1_llama1b_inference_prefix_tuning_ntoken50_lr5e-3_1_epoch2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1208_0_llama1b_ttt_prefix_tuning_lr5e-3_1_token3 \
    --num_virtual_tokens 50 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format