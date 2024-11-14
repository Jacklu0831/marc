# python make_sbatch.py --time 48 --bash_files bash_commands/1114_1_llama1b_predict.sh
# just raw meta llama1b for prediction

# inference with llama1b loras
python predict.py \
    --experiment_folder inference_outputs/experiments/1111_llama8b_inference_with_pretrained_ft_and_lora \
    --pretrained_checkpoint downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --lora_checkpoints_folder downloaded_models/ekinakyurek/marc-lora-adapters-8B-finetuned-llama3 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --max_lora_rank=128 \
    --include_n=1 \
    --new_format
