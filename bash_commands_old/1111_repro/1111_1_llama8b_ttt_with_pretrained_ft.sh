# python make_sbatch.py --time 48 --bash_files bash_commands/1111_1_llama8b_ttt_with_pretrained_ft.sh
# reproduce ttt paper's llama8b performance with pretrained finetune model at https://huggingface.co/ekinakyurek/marc-8B-finetuned-llama3 and lora adapters at  https://huggingface.co/ekinakyurek/marc-lora-adapters-8B-finetuned-llama3

# ttt with pretrained ft lr5e-5
python test_time_train.py \
    --lora_config configs/ttt/8B_lora_single_device.yaml \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1111_llama8b_ttt_with_pretrained_ft \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 2 \
    --lora_rank 128 \
    --lora_alpha 16.0 \
    --learning_rate 5e-5 \
    --new_format

# 53505314