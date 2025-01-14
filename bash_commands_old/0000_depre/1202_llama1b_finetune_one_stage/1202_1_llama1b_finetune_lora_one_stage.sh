# python make_sbatch.py --time 48 --bash_files bash_commands/1202_llama1b_finetune_one_stage/1202_1_llama1b_finetune_lora_one_stage.sh
# ntoken25, lora config same as paper
# for training: lr [5e-3, 1e-3, 5e-4, 1e-4, 5e-5]
# for prediction: epoch2 lr5e-3 based on past experiment

# finetune lora one stage lr5e-3
python finetune_one_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1202_1_llama1b_finetune_lora_one_stage_lr5e-3 \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --inner_epochs 2 \
    --outer_epochs 100 \
    --flash_attn \
    --float16 \
    --learning_rate 5e-3 \
    --use_lora

# finetune lora one stage lr1e-3
python finetune_one_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1202_1_llama1b_finetune_lora_one_stage_lr1e-3 \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --inner_epochs 2 \
    --outer_epochs 100 \
    --flash_attn \
    --float16 \
    --learning_rate 1e-3 \
    --use_lora

# finetune lora one stage lr5e-4
python finetune_one_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1202_1_llama1b_finetune_lora_one_stage_lr5e-4 \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --inner_epochs 2 \
    --outer_epochs 100 \
    --flash_attn \
    --float16 \
    --learning_rate 5e-4 \
    --use_lora

# finetune lora one stage lr1e-4
python finetune_one_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1202_1_llama1b_finetune_lora_one_stage_lr1e-4 \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --inner_epochs 2 \
    --outer_epochs 100 \
    --flash_attn \
    --float16 \
    --learning_rate 1e-4 \
    --use_lora

# finetune lora one stage lr5e-5
python finetune_one_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1202_1_llama1b_finetune_lora_one_stage_lr5e-5 \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --inner_epochs 2 \
    --outer_epochs 100 \
    --flash_attn \
    --float16 \
    --learning_rate 5e-5 \
    --use_lora

# (recursion depth exceeded?)
# Submitted batch job 54154581
# Submitted batch job 54154582
# Submitted batch job 54154583
# Submitted batch job 54154584
# Submitted batch job 54154585

# num worker set to 8, float16, still depth exceeded
# Submitted batch job 54185113
# Submitted batch job 54185114
# Submitted batch job 54185115
# Submitted batch job 54185116
# Submitted batch job 54185117

# save model forward
# Submitted batch job 54213369
# Submitted batch job 54213370
# Submitted batch job 54213371
# Submitted batch job 54213372
# Submitted batch job 54213373

# fix copy_() (just cancelled in order to investigate float16)
# Submitted batch job 54227339
# Submitted batch job 54227340
# Submitted batch job 54227341
# Submitted batch job 54227342
# Submitted batch job 54227343