# python make_sbatch.py --time 48 --bash_files bash_commands/1202_llama1b_finetune_single_stage_lr5e-3_float16/1202_0_llama1b_finetune_single_stage_float16.sh
# ntoken25, lora config same as paper
# for training: lr [5e-3, 1e-3, 5e-4, 1e-4, 5e-5]
# for prediction: epoch4 lr5e-3 based on past experiment

# finetune on train lr5e-3 float16
python finetune_single_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1202_0_llama1b_finetune_single_stage_float16_lr5e-3 \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --inner_epochs 2 \
    --outer_epochs 100 \
    --learning_rate 5e-3 \
    --float16

# finetune on train lr1e-3 float16
python finetune_single_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1202_0_llama1b_finetune_single_stage_float16_lr1e-3 \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --inner_epochs 2 \
    --outer_epochs 100 \
    --learning_rate 1e-3 \
    --float16

# finetune on train lr5e-4 float16
python finetune_single_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1202_0_llama1b_finetune_single_stage_float16_lr5e-4 \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --inner_epochs 2 \
    --outer_epochs 100 \
    --learning_rate 5e-4 \
    --float16

# finetune on train lr1e-4 float16
python finetune_single_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1202_0_llama1b_finetune_single_stage_float16_lr1e-4 \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --inner_epochs 2 \
    --outer_epochs 100 \
    --learning_rate 1e-4 \
    --float16

# finetune on train lr5e-5 float16
python finetune_single_stage.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1202_0_llama1b_finetune_single_stage_float16_lr5e-5 \
    --data_file kaggle_dataset/arc-agi_training_combined.json \
    --new_format \
    --inner_epochs 2 \
    --outer_epochs 100 \
    --learning_rate 5e-5 \
    --float16

# Submitted batch job 54134104
# Submitted batch job 54134105
# Submitted batch job 54134106
# Submitted batch job 54134107
# Submitted batch job 54134108