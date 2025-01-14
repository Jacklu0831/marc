# python make_sbatch.py --time 48 --bash_files bash_commands/1115_0_llama8b_inference_w_wo_lora.sh
# reproduce ttt paper's llama8b performance with no lora or new lora adapters from bash_commands/1114_3_llama8b_ttt.sh

# inference with no lora
python predict.py \
    --experiment_folder inference_outputs/experiments/1115_0_llama8b_inference_no_lora \
    --pretrained_checkpoint downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --max_lora_rank=128 \
    --include_n=1 \
    --new_format

# inference with new lora
python predict.py \
    --experiment_folder inference_outputs/experiments/1115_0_llama8b_inference_and_lora \
    --pretrained_checkpoint downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct \
    --lora_checkpoints_folder train_outputs/1114_llama8b_ttt \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --max_lora_rank=128 \
    --include_n=1 \
    --new_format

# Submitted batch job 53646590
# Submitted batch job 53646591

# Submitted batch job 53646922 # 18
# Submitted batch job 53646923 # 1