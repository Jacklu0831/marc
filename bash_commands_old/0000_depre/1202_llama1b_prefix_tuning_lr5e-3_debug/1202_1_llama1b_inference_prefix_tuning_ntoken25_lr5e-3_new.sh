# python make_sbatch.py --time 48 --bash_files bash_commands/1202_llama1b_prefix_tuning_lr5e-3_debug/1202_1_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_new.sh

# ntoken25 epoch2 new old
python predict_prefix_tuning_old.py \
    --experiment_folder inference_outputs/experiments/1202_0_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_new_old_epoch2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1202_1_llama1b_ttt_prefix_tuning_lr5e-3_new_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected_easy.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected_easy.json \
    --include_n=1 \
    --new_format

# ntoken25 epoch2 new new
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1202_0_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_new_new_epoch2 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1202_1_llama1b_ttt_prefix_tuning_lr5e-3_new_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected_easy.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected_easy.json \
    --include_n=1 \
    --new_format

# Submitted batch job 54198710 # 0/20
# Submitted batch job 54198711 # 0/20