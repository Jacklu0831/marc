# python make_sbatch.py --time 48 --bash_files bash_commands/1216_llama8b_debug/1217_0_llama8b_ttt_epoch0.sh

# llama8b ttt search lr1e-1 epoch2 ntoken1
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1217_0_llama8b_ttt_epoch0 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --num_virtual_tokens 1 \
    --batch_size 1 \
    --grad_accum 2 \
    --learning_rate 0.0 \
    --epochs 1 \
    --new_format \
    --flash_attn \
    --num_max_per_task 1

# Submitted batch job 55207105