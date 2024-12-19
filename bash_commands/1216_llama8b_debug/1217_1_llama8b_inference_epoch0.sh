# python make_sbatch.py --time 48 --bash_files bash_commands/1216_llama8b_debug/1217_1_llama8b_inference_epoch0.sh

# llama8b inference epoch0
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1217_0_llama8b_inference_epoch0 \
    --pretrained_checkpoint downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --pt_checkpoints_folder train_outputs/1217_0_llama8b_ttt_epoch0 \
    --num_virtual_tokens 1 \
    --pt_epoch 1 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --limit_tokens

# Submitted batch job 55144779