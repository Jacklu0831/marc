# python make_sbatch.py --time 48 --bash_files bash_commands/1202_llama1b_finetune_two_stage/1202_1_llama1b_finetune_lora_two_stage.sh
# ntoken25, lora config same as paper
# for training: lora lr [5e-4, 1e-4, 5e-5, 1e-5]
# for prediction: epoch2 lr5e-3 based on past experiment

# finetune lora two stage netlr5e-4
python finetune_two_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1202_0_llama1b_finetune_lora_two_stage_netlr5e-4 \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --prefix_epochs 2 \
    --net_epochs 2 \
    --outer_epochs 100 \
    --prefix_lr 5e-3 \
    --net_lr 5e-4 \
    --flash_attn \
    --float16 \
    --use_lora

# finetune lora two stage netlr1e-4
python finetune_two_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1202_0_llama1b_finetune_lora_two_stage_netlr1e-4 \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --prefix_epochs 2 \
    --net_epochs 2 \
    --outer_epochs 100 \
    --prefix_lr 5e-3 \
    --net_lr 1e-4 \
    --flash_attn \
    --float16 \
    --use_lora

# finetune lora two stage netlr5e-5
python finetune_two_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1202_0_llama1b_finetune_lora_two_stage_netlr5e-5 \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --prefix_epochs 2 \
    --net_epochs 2 \
    --outer_epochs 100 \
    --prefix_lr 5e-3 \
    --net_lr 5e-5 \
    --flash_attn \
    --float16 \
    --use_lora

# finetune lora two stage netlr1e-5
python finetune_two_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1202_0_llama1b_finetune_lora_two_stage_netlr1e-5 \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --prefix_epochs 2 \
    --net_epochs 2 \
    --outer_epochs 100 \
    --prefix_lr 5e-3 \
    --net_lr 1e-5 \
    --flash_attn \
    --float16 \
    --use_lora

# (recursion depth exceeded?)
# Submitted batch job 54154552
# Submitted batch job 54154553
# Submitted batch job 54154554
# Submitted batch job 54154555

# num worker set to 8, float16, still depth exceeded
# Submitted batch job 54185055
# Submitted batch job 54185056
# Submitted batch job 54185057
# Submitted batch job 54185058

# saving model forward
# Submitted batch job 54213333
# Submitted batch job 54213334
# Submitted batch job 54213335
# Submitted batch job 54213336

# fix copy_()
# killed because float16 is bad
# Submitted batch job 54227330
# Submitted batch job 54227331
# Submitted batch job 54227332
# Submitted batch job 54227333