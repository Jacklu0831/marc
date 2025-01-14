# python make_sbatch.py --time 48 --bash_files bash_commands/1126_llama1b_prefix_tuning_lr5e-3/1201_0_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_train.sh

# ntoken25 epoch1 train
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1201_0_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_train_epoch1 \
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
    --train_mode

# ntoken25 epoch3 train
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1201_0_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_train_epoch3 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1126_0_llama1b_ttt_prefix_tuning_lr5e-3_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 3 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --train_mode

# ntoken25 epoch5 train
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1201_0_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_train_epoch5 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1126_0_llama1b_ttt_prefix_tuning_lr5e-3_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 5 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --train_mode

# ntoken25 epoch7 train
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1201_0_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_train_epoch7 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1126_0_llama1b_ttt_prefix_tuning_lr5e-3_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 7 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --train_mode

# ntoken25 epoch9 train
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1201_0_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_train_epoch9 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1126_0_llama1b_ttt_prefix_tuning_lr5e-3_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 9 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --train_mode

# Submitted batch job 54133909 # 8
# Submitted batch job 54133910 # 10
# Submitted batch job 54133911 # 10
# Submitted batch job 54133912 # 9
# Submitted batch job 54133913 # 12