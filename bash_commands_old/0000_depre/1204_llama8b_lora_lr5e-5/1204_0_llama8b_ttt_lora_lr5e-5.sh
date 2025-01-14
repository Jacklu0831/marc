# python make_sbatch.py --time 48 --bash_files bash_commands/1204_llama1b_lora_lr5e-3/1204_0_llama1b_ttt_lora_lr5e-3.sh

# ttt prefix tuning with llama1b ntoken25 float32
python test_time_train_lora.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/test \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 1 \
    --epochs 1 \
    --learning_rate 5e-5 \
    --new_format \
    --float16 \
    --num_tasks 50 \
    --num_max_per_task 250 \
    --logging_steps 1