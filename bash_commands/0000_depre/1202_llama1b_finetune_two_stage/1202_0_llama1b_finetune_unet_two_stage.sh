# python make_sbatch.py --time 48 --bash_files bash_commands/1202_llama1b_finetune_two_stage/1202_0_llama1b_finetune_unet_two_stage.sh
# ntoken25, unet
# for training: unet lr [1e-4, 5e-5, 1e-5, 5e-6]
# for prediction: epoch2 lr5e-3 based on past experiment

# finetune unet two stage netlr1e-4
python finetune_two_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1202_0_llama1b_finetune_unet_two_stage_netlr1e-4 \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --prefix_epochs 2 \
    --net_epochs 2 \
    --outer_epochs 100 \
    --prefix_lr 5e-3 \
    --net_lr 1e-4 \
    --flash_attn \
    --float16

# finetune unet two stage netlr5e-5
python finetune_two_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1202_0_llama1b_finetune_unet_two_stage_netlr5e-5 \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --prefix_epochs 2 \
    --net_epochs 2 \
    --outer_epochs 100 \
    --prefix_lr 5e-3 \
    --net_lr 5e-5 \
    --flash_attn \
    --float16

# finetune unet two stage netlr1e-5
python finetune_two_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1202_0_llama1b_finetune_unet_two_stage_netlr1e-5 \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --prefix_epochs 2 \
    --net_epochs 2 \
    --outer_epochs 100 \
    --prefix_lr 5e-3 \
    --net_lr 1e-5 \
    --flash_attn \
    --float16

# finetune unet two stage netlr5e-6
python finetune_two_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1202_0_llama1b_finetune_unet_two_stage_netlr5e-6 \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --prefix_epochs 2 \
    --net_epochs 2 \
    --outer_epochs 100 \
    --prefix_lr 5e-3 \
    --net_lr 5e-6 \
    --flash_attn \
    --float16

# (recursion depth exceeded?)
# Submitted batch job 54154548
# Submitted batch job 54154549
# Submitted batch job 54154550
# Submitted batch job 54154551

# num worker set to 8, float16, still depth exceeded
# Submitted batch job 54185050
# Submitted batch job 54185051
# Submitted batch job 54185052
# Submitted batch job 54185053

# saving model forward
# Submitted batch job 54213329
# Submitted batch job 54213330
# Submitted batch job 54213331
# Submitted batch job 54213332

# fix copy_()
# killed because float16 is bad
# Submitted batch job 54227270
# Submitted batch job 54227271
# Submitted batch job 54227272
# Submitted batch job 54227273