# python make_sbatch.py --time 48 --bash_files bash_commands/1214_llama1b_lora/1214_1_llama1b_inference.sh

# inference lora with llama1b
python predict.py \
    --experiment_folder inference_outputs/experiments/1214_1_llama1b_inference_lora \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --lora_checkpoints_folder train_outputs/1214_0_llama1b_ttt_lora_train \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_training_challenges.json \
    --solution_file kaggle_dataset/arc-agi_training_solutions.json \
    --max_lora_rank=128 \
    --include_n=1 \
    --new_format

# Submitted batch job 54877947