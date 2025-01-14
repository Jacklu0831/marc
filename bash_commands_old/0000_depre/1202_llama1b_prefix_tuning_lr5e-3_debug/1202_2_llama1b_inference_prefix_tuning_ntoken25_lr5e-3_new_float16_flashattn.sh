# python make_sbatch.py --time 48 --bash_files bash_commands/1202_llama1b_prefix_tuning_lr5e-3_debug/1202_0_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_new_float16_flashattn.sh

# ntoken25 epoch2 old new float16
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/test \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1202_0_llama1b_ttt_prefix_tuning_lr5e-3_old_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected_easy.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected_easy.json \
    --include_n=1 \
    --new_format \
    --float16

# ntoken25 epoch2 old new flashattn
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/test \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1202_0_llama1b_ttt_prefix_tuning_lr5e-3_old_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected_easy.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected_easy.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# ntoken25 epoch2 old new float16 flashattn
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/test \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1202_0_llama1b_ttt_prefix_tuning_lr5e-3_old_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected_easy.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected_easy.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --float16

# Submitted batch job 54220410
# Submitted batch job 54220411
# Submitted batch job 54220412