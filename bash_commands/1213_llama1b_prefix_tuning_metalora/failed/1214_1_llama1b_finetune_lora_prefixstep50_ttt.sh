# python make_sbatch.py --time 48 --bash_files bash_commands/1213_llama1b_prefix_tuning_metalora/1214_1_llama1b_finetune_lora_prefixstep50_ttt.sh

# ttt prefix tuning 50prefixstep loraepoch1
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1214_1_llama1b_finetune_lora_prefixstep50_ttt_loraepoch1 \
    --lora_ckpt train_outputs/1214_0_llama1b_finetune_lora_two_stage_50prefixstep/checkpoint-outer-epoch1.pt \
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

# ttt prefix tuning 50prefixstep loraepoch3
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1214_1_llama1b_finetune_lora_prefixstep50_ttt_loraepoch3 \
    --lora_ckpt train_outputs/1214_0_llama1b_finetune_lora_two_stage_50prefixstep/checkpoint-outer-epoch3.pt \
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

# ttt prefix tuning 50prefixstep loraepoch6
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1214_1_llama1b_finetune_lora_prefixstep50_ttt_loraepoch6 \
    --lora_ckpt train_outputs/1214_0_llama1b_finetune_lora_two_stage_50prefixstep/checkpoint-outer-epoch6.pt \
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

# Submitted batch job 54947072
# Submitted batch job 54947073
# Submitted batch job 54947074