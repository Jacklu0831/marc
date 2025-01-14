# python make_sbatch.py --time 48 --bash_files bash_commands/1213_llama1b_prefix_tuning_metalora/1216_3_llama1b_finetune_lora_two_stage_0prefixstep.sh

# finetune lora two stage 1prefixstep
python finetune_two_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1216_3_llama1b_finetune_lora_two_stage_0prefixstep \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --num_virtual_tokens 1 \
    --prefix_epochs 0 \
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
    --cache_dataset \
    --wandb

# Submitted batch job 55162485