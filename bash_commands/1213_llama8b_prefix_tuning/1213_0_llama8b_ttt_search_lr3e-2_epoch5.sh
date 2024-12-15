# python make_sbatch.py --time 48 --bash_files bash_commands/1213_llama8b_prefix_tuning/1213_0_llama8b_ttt_search_lr3e-2_epoch5.sh

# llama8b ttt search lr3e-2 epoch5 ntoken1
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1213_0_llama8b_ttt_search_lr3e-2_epoch5_ntoken1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 1 \
    --grad_accum 2 \
    --epochs 5 \
    --num_virtual_tokens 1 \
    --learning_rate 3e-2 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb

# llama8b ttt search lr3e-2 epoch5 ntoken5
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1213_0_llama8b_ttt_search_lr3e-2_epoch5_ntoken5 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 1 \
    --grad_accum 2 \
    --epochs 5 \
    --num_virtual_tokens 5 \
    --learning_rate 3e-2 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb

# llama8b ttt search lr3e-2 epoch5 ntoken10
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1213_0_llama8b_ttt_search_lr3e-2_epoch5_ntoken10 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 1 \
    --grad_accum 2 \
    --epochs 5 \
    --num_virtual_tokens 10 \
    --learning_rate 3e-2 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb

# llama8b ttt search lr3e-2 epoch5 ntoken50
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1213_0_llama8b_ttt_search_lr3e-2_epoch5_ntoken50 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 1 \
    --grad_accum 2 \
    --epochs 5 \
    --num_virtual_tokens 50 \
    --learning_rate 3e-2 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb

# Submitted batch job 54878055
# Submitted batch job 54878056
# Submitted batch job 54878057
# Submitted batch job 54878058