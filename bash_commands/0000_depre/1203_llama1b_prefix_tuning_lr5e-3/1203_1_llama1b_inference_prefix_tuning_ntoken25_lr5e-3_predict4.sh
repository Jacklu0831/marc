# python make_sbatch.py --time 48 --bash_files bash_commands/1203_llama1b_prefix_tuning_lr5e-3/1203_1_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_predict4.sh

# ntoken25 epoch2 float32 predict4
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1203_1_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_float32_epoch2_predict4 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1203_0_llama1b_ttt_prefix_tuning_lr5e-3_float32_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --float16

# ntoken25 epoch4 float32 predict4
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1203_1_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_float32_epoch4_predict4 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1203_0_llama1b_ttt_prefix_tuning_lr5e-3_float32_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 4 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --float16





# ntoken25 epoch2 float16 predict4
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1203_1_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_float16_epoch2_predict4 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1203_0_llama1b_ttt_prefix_tuning_lr5e-3_float16_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --float16

# ntoken25 epoch4 float16 predict4
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1203_1_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_float16_epoch4_predict4 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1203_0_llama1b_ttt_prefix_tuning_lr5e-3_float16_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 4 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --float16







# ntoken25 epoch2 float32 flashattn predict4
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1203_1_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_float32_flashattn_epoch2_predict4 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1203_0_llama1b_ttt_prefix_tuning_lr5e-3_float32_flashattn_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --float16

# ntoken25 epoch4 float32 flashattn predict4
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1203_1_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_float32_flashattn_epoch4_predict4 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1203_0_llama1b_ttt_prefix_tuning_lr5e-3_float32_flashattn_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 4 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --float16






# ntoken25 epoch2 float16 flashattn predict4
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1203_1_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_float16_flashattn_epoch2_predict4 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1203_0_llama1b_ttt_prefix_tuning_lr5e-3_float16_flashattn_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 2 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --float16

# ntoken25 epoch4 float16 flashattn predict4
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1203_1_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_float16_flashattn_epoch4_predict4 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1203_0_llama1b_ttt_prefix_tuning_lr5e-3_float16_flashattn_ntoken25 \
    --num_virtual_tokens 25 \
    --pt_epoch 4 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --float16

# Submitted batch job 54238131
# Submitted batch job 54238132
# Submitted batch job 54238133
# Submitted batch job 54238134
# Submitted batch job 54238135
# Submitted batch job 54238136
# Submitted batch job 54238137
# Submitted batch job 54238138