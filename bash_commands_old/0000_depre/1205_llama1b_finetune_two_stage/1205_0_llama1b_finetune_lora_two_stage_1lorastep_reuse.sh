# python make_sbatch.py --time 48 --bash_files bash_commands/1205_llama1b_finetune_two_stage/1205_0_llama1b_finetune_lora_two_stage_1lorastep_reuse.sh

# finetune lora two stage 1prefixstep
python finetune_two_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1205_0_llama1b_finetune_lora_two_stage_1prefixstep_1lorastep_reuse \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --num_virtual_tokens 25 \
    --prefix_steps 1 \
    --net_steps 1 \
    --outer_epochs 100 \
    --prefix_lr 5e-3 \
    --net_lr 5e-5 \
    --flash_attn \
    --extra_leave_n 1 \
    --use_lora \
    --reuse_prefix \
    --pt_checkpoints_folder train_outputs/1205_0_llama1b_ttt_prefix_tuning_traincombined_lr5e-3 \
    --pt_epoch 2 \
    --save_every 1

# finetune lora two stage 5prefixstep
python finetune_two_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1205_0_llama1b_finetune_lora_two_stage_5prefixstep_1lorastep_reuse \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --num_virtual_tokens 25 \
    --prefix_steps 5 \
    --net_steps 1 \
    --outer_epochs 100 \
    --prefix_lr 5e-3 \
    --net_lr 5e-5 \
    --flash_attn \
    --extra_leave_n 1 \
    --use_lora \
    --reuse_prefix \
    --pt_checkpoints_folder train_outputs/1205_0_llama1b_ttt_prefix_tuning_traincombined_lr5e-3 \
    --pt_epoch 2 \
    --save_every 1

# finetune lora two stage 10prefixstep
python finetune_two_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1205_0_llama1b_finetune_lora_two_stage_10prefixstep_1lorastep_reuse \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --num_virtual_tokens 25 \
    --prefix_steps 10 \
    --net_steps 1 \
    --outer_epochs 100 \
    --prefix_lr 5e-3 \
    --net_lr 5e-5 \
    --flash_attn \
    --extra_leave_n 1 \
    --use_lora \
    --reuse_prefix \
    --pt_checkpoints_folder train_outputs/1205_0_llama1b_ttt_prefix_tuning_traincombined_lr5e-3 \
    --pt_epoch 2 \
    --save_every 1

# finetune lora two stage 25prefixstep
python finetune_two_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1205_0_llama1b_finetune_lora_two_stage_25prefixstep_1lorastep_reuse \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --num_virtual_tokens 25 \
    --prefix_steps 25 \
    --net_steps 1 \
    --outer_epochs 100 \
    --prefix_lr 5e-3 \
    --net_lr 5e-5 \
    --flash_attn \
    --extra_leave_n 1 \
    --use_lora \
    --reuse_prefix \
    --pt_checkpoints_folder train_outputs/1205_0_llama1b_ttt_prefix_tuning_traincombined_lr5e-3 \
    --pt_epoch 2 \
    --save_every 1

# finetune lora two stage 50prefixstep
python finetune_two_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1205_0_llama1b_finetune_lora_two_stage_50prefixstep_1lorastep_reuse \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --num_virtual_tokens 25 \
    --prefix_steps 50 \
    --net_steps 1 \
    --outer_epochs 100 \
    --prefix_lr 5e-3 \
    --net_lr 5e-5 \
    --flash_attn \
    --extra_leave_n 1 \
    --use_lora \
    --reuse_prefix \
    --pt_checkpoints_folder train_outputs/1205_0_llama1b_ttt_prefix_tuning_traincombined_lr5e-3 \
    --pt_epoch 2 \
    --save_every 1

# Submitted batch job 54572913
# Submitted batch job 54572914
# Submitted batch job 54572915
# Submitted batch job 54572916
# Submitted batch job 54572917
# 50 steps takes <3 hours per outer epoch