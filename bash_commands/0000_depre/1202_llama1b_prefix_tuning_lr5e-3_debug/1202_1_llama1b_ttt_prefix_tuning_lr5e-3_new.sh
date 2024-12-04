# python make_sbatch.py --time 48 --bash_files bash_commands/1202_llama1b_prefix_tuning_lr5e-3_debug/1202_1_llama1b_ttt_prefix_tuning_lr5e-3_new.sh
# train ttt with raw meta llama1b, can ablate lora_to_output

# ttt prefix tuning with llama1b ntoken25 new
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1202_1_llama1b_ttt_prefix_tuning_lr5e-3_new_ntoken25 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected_easy.json \
    --batch_size 2 \
    --epochs 2 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format

# Submitted batch job 54197785