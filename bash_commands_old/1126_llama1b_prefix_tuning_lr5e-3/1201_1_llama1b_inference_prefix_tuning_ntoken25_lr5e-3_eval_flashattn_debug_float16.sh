# python make_sbatch.py --time 48 --bash_files bash_commands/1126_llama1b_prefix_tuning_lr5e-3/1201_1_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_eval_flashattn_debug_float16.sh

# ntoken25 epoch1 eval debug float16
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1201_0_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_eval_epoch1_flashattn_debug_float16 \
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
    --flash_attn \
    --float16

# ntoken25 epoch3 eval debug float16
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1201_0_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_eval_epoch3_flashattn_debug_float16 \
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
    --flash_attn \
    --float16

# ntoken25 epoch5 eval debug float16
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1201_0_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_eval_epoch5_flashattn_debug_float16 \
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
    --flash_attn \
    --float16

# ntoken25 epoch7 eval debug float16
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1201_0_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_eval_epoch7_flashattn_debug_float16 \
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
    --flash_attn \
    --float16

# ntoken25 epoch9 eval debug float16
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1201_0_llama1b_inference_prefix_tuning_ntoken25_lr5e-3_eval_epoch9_flashattn_debug_float16 \
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
    --flash_attn \
    --float16

# Submitted batch job 54194830
# Submitted batch job 54194831
# Submitted batch job 54194832
# Submitted batch job 54194833
# Submitted batch job 54194834