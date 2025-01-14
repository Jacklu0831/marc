# python make_sbatch.py --time 48 --bash_files bash_commands/1114_2_llama8b_inference_with_pretrained_ft_w_wo_lora.sh
# reproduce ttt paper's llama8b performance with pretrained finetune model at https://huggingface.co/ekinakyurek/marc-8B-finetuned-llama3
# and no lora or new lora adapters from bash_commands/1111_1_llama8b_ttt_with_pretrained_ft.sh

# inference with pretrained ft and no lora
python predict.py \
    --experiment_folder inference_outputs/experiments/1114_2_llama8b_inference_with_pretrained_ft_and_no_lora \
    --pretrained_checkpoint downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --max_lora_rank=128 \
    --include_n=1 \
    --new_format

# inference with pretrained ft and new lora
python predict.py \
    --experiment_folder inference_outputs/experiments/1114_2_llama8b_inference_with_pretrained_ft_and_new_lora \
    --pretrained_checkpoint downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --lora_checkpoints_folder train_outputs/1111_llama8b_ttt_with_pretrained_ft \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --max_lora_rank=128 \
    --include_n=1 \
    --new_format

# Submitted batch job 53553754 # 16
# Submitted batch job 53553753 # 32