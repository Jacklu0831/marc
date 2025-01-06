# python make_sbatch.py --time 48 --bash_files bash_commands/1216_llama8b_debug/1217_0_llama8b_ttt_reuseprefix.sh

# ttt prefix tuning with llama8b ntoken1
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1217_0_llama8b_ttt_reuseprefix_lr5e-3_ntoken1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 1 \
    --grad_accum 2 \
    --epochs 10 \
    --num_virtual_tokens 1 \
    --learning_rate 5e-3 \
    --new_format \
    --flash_attn \
    --wandb \
    --reuse_prefix

# Submitted batch job 55072969