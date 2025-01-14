# python make_sbatch.py --time 48 --bash_files bash_commands/1111_0_llama8b_inference_with_pretrained_ft_and_lora.sh
# reproduce ttt paper's llama1b performance with pretrained finetune model at https://huggingface.co/ekinakyurek/marc-8B-finetuned-llama3 and lora adapters at  https://huggingface.co/ekinakyurek/marc-lora-adapters-8B-finetuned-llama3

# inference with pretrained ft and lora
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

# 53441508 # 33