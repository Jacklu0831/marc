# python make_sbatch.py --time 48 --bash_files bash_commands/1213_llama1b_prefix_tuning_metalora/1216_0_llama1b_finetune_lora_two_stage.sh

# finetune lora two stage 1prefixstep
python finetune_two_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1216_0_llama1b_finetune_lora_two_stage_1prefixstep \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --num_virtual_tokens 1 \
    --prefix_steps 1 \
    --net_steps 1 \
    --outer_epochs 100 \
    --prefix_lr 1e-1 \
    --net_lr 5e-5 \
    --flash_attn \
    --extra_leave_n 1 \
    --use_lora \
    --reuse_prefix \
    --pt_checkpoints_folder train_outputs/1213_0_ttt_lr1e-1_epoch20_ntoken1_traincombined \
    --pt_epoch 20 \
    --wandb

# finetune lora two stage 5prefixstep
python finetune_two_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1216_0_llama1b_finetune_lora_two_stage_5prefixstep \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --num_virtual_tokens 1 \
    --prefix_steps 5 \
    --net_steps 1 \
    --outer_epochs 100 \
    --prefix_lr 1e-1 \
    --net_lr 5e-5 \
    --flash_attn \
    --extra_leave_n 1 \
    --use_lora \
    --reuse_prefix \
    --pt_checkpoints_folder train_outputs/1213_0_ttt_lr1e-1_epoch20_ntoken1_traincombined \
    --pt_epoch 20 \
    --wandb

# finetune lora two stage 10prefixstep
python finetune_two_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1216_0_llama1b_finetune_lora_two_stage_10prefixstep \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --num_virtual_tokens 1 \
    --prefix_steps 10 \
    --net_steps 1 \
    --outer_epochs 100 \
    --prefix_lr 1e-1 \
    --net_lr 5e-5 \
    --flash_attn \
    --extra_leave_n 1 \
    --use_lora \
    --reuse_prefix \
    --pt_checkpoints_folder train_outputs/1213_0_ttt_lr1e-1_epoch20_ntoken1_traincombined \
    --pt_epoch 20 \
    --wandb

# finetune lora two stage 50prefixstep
python finetune_two_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1216_0_llama1b_finetune_lora_two_stage_50prefixstep \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --num_virtual_tokens 1 \
    --prefix_steps 50 \
    --net_steps 1 \
    --outer_epochs 100 \
    --prefix_lr 1e-1 \
    --net_lr 5e-5 \
    --flash_attn \
    --extra_leave_n 1 \
    --use_lora \
    --reuse_prefix \
    --pt_checkpoints_folder train_outputs/1213_0_ttt_lr1e-1_epoch20_ntoken1_traincombined \
    --pt_epoch 20 \
    --wandb

# Submitted batch job 54940337
# Submitted batch job 54940339
# Submitted batch job 54940340
# Submitted batch job 54940341