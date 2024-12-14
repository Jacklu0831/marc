# python make_sbatch.py --time 48 --bash_files bash_commands/1213_llama1b_prefix_tuning_metalora/1213_0_ttt_lr1e-1_epoch20_ntoken1_train.sh

# ttt lr1e-1 epoch20 ntoken1 train split0-10
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1213_0_ttt_lr1e-1_epoch20_ntoken1_train \
    --data_file kaggle_dataset/arc-agi_training_challenges.json \
    --batch_size 2 \
    --epochs 20 \
    --num_virtual_tokens 1 \
    --learning_rate 1e-1 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb \
    --div 10 \
    --mod 0

# ttt lr1e-1 epoch20 ntoken1 train split1-10
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1213_0_ttt_lr1e-1_epoch20_ntoken1_train \
    --data_file kaggle_dataset/arc-agi_training_challenges.json \
    --batch_size 2 \
    --epochs 20 \
    --num_virtual_tokens 1 \
    --learning_rate 1e-1 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb \
    --div 10 \
    --mod 1

# ttt lr1e-1 epoch20 ntoken1 train split2-10
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1213_0_ttt_lr1e-1_epoch20_ntoken1_train \
    --data_file kaggle_dataset/arc-agi_training_challenges.json \
    --batch_size 2 \
    --epochs 20 \
    --num_virtual_tokens 1 \
    --learning_rate 1e-1 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb \
    --div 10 \
    --mod 2

# ttt lr1e-1 epoch20 ntoken1 train split3-10
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1213_0_ttt_lr1e-1_epoch20_ntoken1_train \
    --data_file kaggle_dataset/arc-agi_training_challenges.json \
    --batch_size 2 \
    --epochs 20 \
    --num_virtual_tokens 1 \
    --learning_rate 1e-1 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb \
    --div 10 \
    --mod 3

# ttt lr1e-1 epoch20 ntoken1 train split4-10
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1213_0_ttt_lr1e-1_epoch20_ntoken1_train \
    --data_file kaggle_dataset/arc-agi_training_challenges.json \
    --batch_size 2 \
    --epochs 20 \
    --num_virtual_tokens 1 \
    --learning_rate 1e-1 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb \
    --div 10 \
    --mod 4

# ttt lr1e-1 epoch20 ntoken1 train split5-10
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1213_0_ttt_lr1e-1_epoch20_ntoken1_train \
    --data_file kaggle_dataset/arc-agi_training_challenges.json \
    --batch_size 2 \
    --epochs 20 \
    --num_virtual_tokens 1 \
    --learning_rate 1e-1 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb \
    --div 10 \
    --mod 5

# ttt lr1e-1 epoch20 ntoken1 train split6-10
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1213_0_ttt_lr1e-1_epoch20_ntoken1_train \
    --data_file kaggle_dataset/arc-agi_training_challenges.json \
    --batch_size 2 \
    --epochs 20 \
    --num_virtual_tokens 1 \
    --learning_rate 1e-1 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb \
    --div 10 \
    --mod 6

# ttt lr1e-1 epoch20 ntoken1 train split7-10
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1213_0_ttt_lr1e-1_epoch20_ntoken1_train \
    --data_file kaggle_dataset/arc-agi_training_challenges.json \
    --batch_size 2 \
    --epochs 20 \
    --num_virtual_tokens 1 \
    --learning_rate 1e-1 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb \
    --div 10 \
    --mod 7

# ttt lr1e-1 epoch20 ntoken1 train split8-10
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1213_0_ttt_lr1e-1_epoch20_ntoken1_train \
    --data_file kaggle_dataset/arc-agi_training_challenges.json \
    --batch_size 2 \
    --epochs 20 \
    --num_virtual_tokens 1 \
    --learning_rate 1e-1 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb \
    --div 10 \
    --mod 8

# ttt lr1e-1 epoch20 ntoken1 train split9-10
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1213_0_ttt_lr1e-1_epoch20_ntoken1_train \
    --data_file kaggle_dataset/arc-agi_training_challenges.json \
    --batch_size 2 \
    --epochs 20 \
    --num_virtual_tokens 1 \
    --learning_rate 1e-1 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb \
    --div 10 \
    --mod 9

# Submitted batch job 54813986
# Submitted batch job 54813987
# Submitted batch job 54813988
# Submitted batch job 54813989
# Submitted batch job 54813990
# Submitted batch job 54813991
# Submitted batch job 54813992
# Submitted batch job 54813993
# Submitted batch job 54813994
# Submitted batch job 54813995