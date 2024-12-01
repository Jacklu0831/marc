python test_time_train.py \
    --lora_config configs/ttt/1B_lora_single_device.yaml \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/test \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 2 \
    --lora_rank 128 \
    --lora_alpha 16.0 \
    --learning_rate 5e-5 \
    --new_format \
    --num_tasks 3


python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/test \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 1 \
    --learning_rate 5e-3 \
    --new_format \
    --num_tasks 3