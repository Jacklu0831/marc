# python make_sbatch.py --time 48 --bash_files bash_commands/1114_3_llama8b_ttt.sh
# train ttt with raw meta llama8b, can ablate lora_to_output

# ttt with llama8b
python test_time_train.py \
    --lora_config configs/ttt/meta8B_lora_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct \
    --experiment_folder train_outputs/1114_llama8b_ttt \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 2 \
    --lora_rank 128 \
    --lora_alpha 16.0 \
    --learning_rate 5e-5 \
    --new_format

# Submitted batch job 53604433