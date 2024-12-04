# python make_sbatch.py --time 48 --bash_files bash_commands/1202_llama1b_finetune_two_stage_reuse/1202_0_llama1b_finetune_unet_two_stage_reuse.sh
# ntoken25, unet
# for training: unet lr [1e-4, 5e-5, 1e-5, 5e-6]
# for prediction: epoch2 lr5e-3 based on past experiment

# finetune unet two stage reuse netlr1e-4
python finetune_two_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1202_0_llama1b_finetune_unet_two_stage_reuse_netlr1e-4 \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --start_prefix_epochs 2 \
    --prefix_epochs 1 \
    --net_epochs 2 \
    --outer_epochs 100 \
    --prefix_lr 5e-3 \
    --net_lr 1e-4 \
    --flash_attn \
    --float16 \
    --reuse_prefix

# finetune unet two stage reuse netlr5e-5
python finetune_two_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1202_0_llama1b_finetune_unet_two_stage_reuse_netlr5e-5 \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --start_prefix_epochs 2 \
    --prefix_epochs 1 \
    --net_epochs 2 \
    --outer_epochs 100 \
    --prefix_lr 5e-3 \
    --net_lr 5e-5 \
    --flash_attn \
    --float16 \
    --reuse_prefix

# finetune unet two stage reuse netlr1e-5
python finetune_two_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1202_0_llama1b_finetune_unet_two_stage_reuse_netlr1e-5 \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --start_prefix_epochs 2 \
    --prefix_epochs 1 \
    --net_epochs 2 \
    --outer_epochs 100 \
    --prefix_lr 5e-3 \
    --net_lr 1e-5 \
    --flash_attn \
    --float16 \
    --reuse_prefix

# finetune unet two stage reuse netlr5e-6
python finetune_two_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1202_0_llama1b_finetune_unet_two_stage_reuse_netlr5e-6 \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --start_prefix_epochs 2 \
    --prefix_epochs 1 \
    --net_epochs 2 \
    --outer_epochs 100 \
    --prefix_lr 5e-3 \
    --net_lr 5e-6 \
    --flash_attn \
    --float16 \
    --reuse_prefix

# Submitted batch job 54255517
# Submitted batch job 54255518
# Submitted batch job 54255519
# Submitted batch job 54255520