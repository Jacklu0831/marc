# python make_sbatch.py --time 48 --bash_files bash_commands/1114_0_llama1b_ttt.sh
# train ttt with raw meta llama1b

# ttt with llama1b
python test_time_train.py \
    --lora_config configs/ttt/1B_lora_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1114_llama1b_ttt \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 2 \
    --lora_rank 128 \
    --lora_alpha 16.0 \
    --learning_rate 5e-5 \
    --new_format

# 53536033