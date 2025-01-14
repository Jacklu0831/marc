# python make_sbatch.py --time 48 --bash_files bash_commands/1216_llama8b_debug/1216_1_inference_lr1e-1_epoch20_ntoken1_rerunsplit.sh

# inference lr1e-1 epoch20 ntoken1
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1216_1_inference_lr1e-1_epoch20_ntoken1_rerunsplit \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1216_0_ttt_lr1e-1_epoch20_ntoken1_rerunsplit \
    --num_virtual_tokens 1 \
    --pt_epoch 20 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --limit_tokens

# Submitted batch job 55017855 # 9/80
# Submitted batch job 55100587 # 11/80