# python make_sbatch.py --time 48 --bash_files bash_commands/1202_llama1b_finetune_one_stage/1202_0_llama1b_finetune_unet_one_stage.sh
# ntoken25, unet
# for training: lr [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
# for prediction: epoch2 lr5e-3 based on past experiment

# finetune unet one stage lr1e-3
python finetune_one_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1202_0_llama1b_finetune_unet_one_stage_lr1e-3 \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --inner_epochs 2 \
    --outer_epochs 100 \
    --flash_attn \
    --float16 \
    --learning_rate 1e-3

# finetune unet one stage lr5e-4
python finetune_one_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1202_0_llama1b_finetune_unet_one_stage_lr5e-4 \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --inner_epochs 2 \
    --outer_epochs 100 \
    --flash_attn \
    --float16 \
    --learning_rate 5e-4

# finetune unet one stage lr1e-4
python finetune_one_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1202_0_llama1b_finetune_unet_one_stage_lr1e-4 \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --inner_epochs 2 \
    --outer_epochs 100 \
    --flash_attn \
    --float16 \
    --learning_rate 1e-4

# finetune unet one stage lr5e-5
python finetune_one_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1202_0_llama1b_finetune_unet_one_stage_lr5e-5 \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --inner_epochs 2 \
    --outer_epochs 100 \
    --flash_attn \
    --float16 \
    --learning_rate 5e-5

# finetune unet one stage lr1e-5
python finetune_one_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1202_0_llama1b_finetune_unet_one_stage_lr1e-5 \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --inner_epochs 2 \
    --outer_epochs 100 \
    --flash_attn \
    --float16 \
    --learning_rate 1e-5

# (recursion depth exceeded?)
# Submitted batch job 54154576
# Submitted batch job 54154577
# Submitted batch job 54154578
# Submitted batch job 54154579
# Submitted batch job 54154580

# num worker set to 8, float16, still depth exceeded
# Submitted batch job 54185107
# Submitted batch job 54185108
# Submitted batch job 54185109
# Submitted batch job 54185110
# Submitted batch job 54185111

# save model foward
# Submitted batch job 54213363
# Submitted batch job 54213364
# Submitted batch job 54213365
# Submitted batch job 54213366
# Submitted batch job 54213367

# fix copy_() (just cancelled in order to investigate float16)
# Submitted batch job 54227334
# Submitted batch job 54227335
# Submitted batch job 54227336
# Submitted batch job 54227337
# Submitted batch job 54227338