# python make_sbatch.py --time 48 --bash_files bash_commands/1213_llama8b_prefix_tuning/1213_0_llama8b_ttt_search_lr3e-4_epoch20.sh

# llama8b ttt search lr3e-4 epoch20 ntoken1 split0
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1213_0_llama8b_ttt_search_lr3e-4_epoch20_ntoken1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 1 \
    --grad_accum 2 \
    --epochs 20 \
    --num_virtual_tokens 1 \
    --learning_rate 3e-4 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb \
    --div 2 \
    --mod 0

# llama8b ttt search lr3e-4 epoch20 ntoken1 split1
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1213_0_llama8b_ttt_search_lr3e-4_epoch20_ntoken1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 1 \
    --grad_accum 2 \
    --epochs 20 \
    --num_virtual_tokens 1 \
    --learning_rate 3e-4 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb \
    --div 2 \
    --mod 1

# llama8b ttt search lr3e-4 epoch20 ntoken5 split0
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1213_0_llama8b_ttt_search_lr3e-4_epoch20_ntoken5 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 1 \
    --grad_accum 2 \
    --epochs 20 \
    --num_virtual_tokens 5 \
    --learning_rate 3e-4 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb \
    --div 2 \
    --mod 0

# llama8b ttt search lr3e-4 epoch20 ntoken5 split1
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1213_0_llama8b_ttt_search_lr3e-4_epoch20_ntoken5 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 1 \
    --grad_accum 2 \
    --epochs 20 \
    --num_virtual_tokens 5 \
    --learning_rate 3e-4 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb \
    --div 2 \
    --mod 1

# llama8b ttt search lr3e-4 epoch20 ntoken10 split0
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1213_0_llama8b_ttt_search_lr3e-4_epoch20_ntoken10 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 1 \
    --grad_accum 2 \
    --epochs 20 \
    --num_virtual_tokens 10 \
    --learning_rate 3e-4 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb \
    --div 2 \
    --mod 0

# llama8b ttt search lr3e-4 epoch20 ntoken10 split1
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1213_0_llama8b_ttt_search_lr3e-4_epoch20_ntoken10 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 1 \
    --grad_accum 2 \
    --epochs 20 \
    --num_virtual_tokens 10 \
    --learning_rate 3e-4 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb \
    --div 2 \
    --mod 1

# llama8b ttt search lr3e-4 epoch20 ntoken50 split0
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1213_0_llama8b_ttt_search_lr3e-4_epoch20_ntoken50 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 1 \
    --grad_accum 2 \
    --epochs 20 \
    --num_virtual_tokens 50 \
    --learning_rate 3e-4 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb \
    --div 2 \
    --mod 0

# llama8b ttt search lr3e-4 epoch20 ntoken50 split1
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1213_0_llama8b_ttt_search_lr3e-4_epoch20_ntoken50 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 1 \
    --grad_accum 2 \
    --epochs 20 \
    --num_virtual_tokens 50 \
    --learning_rate 3e-4 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb \
    --div 2 \
    --mod 1

# Submitted batch job 55012329
# Submitted batch job 55012330
# Submitted batch job 55012331
# Submitted batch job 55012332
# Submitted batch job 55012333
# Submitted batch job 55012334
# Submitted batch job 55012335
# Submitted batch job 55012336