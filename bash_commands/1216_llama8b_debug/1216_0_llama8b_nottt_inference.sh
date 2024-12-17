# python make_sbatch.py --time 48 --bash_files bash_commands/1216_llama8b_debug/1216_0_llama8b_nottt_inference.sh

# llama8b nottt inference new script
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1216_0_llama8b_nottt_inference_newscript \
    --pretrained_checkpoint downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --limit_tokens

# llama8b nottt inference old script
python predict.py \
    --experiment_folder inference_outputs/experiments/1216_0_llama8b_nottt_inference_oldscript \
    --pretrained_checkpoint downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --max_lora_rank=128 \
    --include_n=1 \
    --new_format
