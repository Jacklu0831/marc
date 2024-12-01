# python make_sbatch.py --time 48 --bash_files bash_commands/1201_llama1b_prefix_tuning_lr5e-3_float16/1201_2_llama1b_inference_prefix_tuning_ntoken25_lr5e-3.sh

# ntoken25 epoch1 float16
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1201_2_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_epoch1 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1201_0_llama1b_ttt_prefix_tuning_lr5e-3_float16_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# ntoken25 epoch2 float16
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1201_2_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_epoch2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1201_0_llama1b_ttt_prefix_tuning_lr5e-3_float16_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# ntoken25 epoch3 float16
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1201_2_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_epoch3 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1201_0_llama1b_ttt_prefix_tuning_lr5e-3_float16_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 3 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format

# ntoken25 epoch4 float16
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1201_2_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_epoch4 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1201_0_llama1b_ttt_prefix_tuning_lr5e-3_float16_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 4 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format
