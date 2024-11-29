# python make_sbatch.py --time 48 --bash_files bash_commands/1114_1_llama1b_inference_w_wo_lora.sh
# just raw meta llama1b for prediction

# inference with no lora
python predict.py \
    --experiment_folder inference_outputs/experiments/1114_1_llama1b_inference_and_no_lora \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --max_lora_rank=128 \
    --include_n=1 \
    --new_format

# inference with new lora
python predict.py \
    --experiment_folder inference_outputs/experiments/1114_1_llama1b_inference_and_new_lora \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --lora_checkpoints_folder train_outputs/1114_llama1b_ttt \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --max_lora_rank=128 \
    --include_n=1 \
    --new_format

# Submitted batch job 53563189 # 0
# Submitted batch job 53563188 # 7