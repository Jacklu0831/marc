# python make_sbatch.py --time 48 --bash_files bash_commands/1213_llama1b_prefix_tuning_metalora/1216_1_llama1b_finetune_lora_prefixstep10_ttt.sh

# ttt prefix tuning 10prefixstep loraepoch5
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1216_1_llama1b_finetune_lora_prefixstep10_ttt_loraepoch5 \
    --lora_ckpt train_outputs/1216_0_llama1b_finetune_lora_two_stage_10prefixstep/checkpoint-outer-epoch5.pt \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 20 \
    --num_virtual_tokens 1 \
    --learning_rate 1e-1 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb

# ttt prefix tuning 10prefixstep loraepoch10
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1216_1_llama1b_finetune_lora_prefixstep10_ttt_loraepoch10 \
    --lora_ckpt train_outputs/1216_0_llama1b_finetune_lora_two_stage_10prefixstep/checkpoint-outer-epoch10.pt \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 20 \
    --num_virtual_tokens 1 \
    --learning_rate 1e-1 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb

# ttt prefix tuning 10prefixstep loraepoch25
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1216_1_llama1b_finetune_lora_prefixstep10_ttt_loraepoch25 \
    --lora_ckpt train_outputs/1216_0_llama1b_finetune_lora_two_stage_10prefixstep/checkpoint-outer-epoch25.pt \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 20 \
    --num_virtual_tokens 1 \
    --learning_rate 1e-1 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb

# ttt prefix tuning 10prefixstep loraepoch50
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1216_1_llama1b_finetune_lora_prefixstep10_ttt_loraepoch50 \
    --lora_ckpt train_outputs/1216_0_llama1b_finetune_lora_two_stage_10prefixstep/checkpoint-outer-epoch50.pt \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 20 \
    --num_virtual_tokens 1 \
    --learning_rate 1e-1 \
    --new_format \
    --flash_attn \
    --lr_scheduler_type cosine \
    --warmup_steps 5 \
    --wandb

# Submitted batch job 55101504
# Submitted batch job 55101505
# Submitted batch job 55101506
# Submitted batch job 55101507