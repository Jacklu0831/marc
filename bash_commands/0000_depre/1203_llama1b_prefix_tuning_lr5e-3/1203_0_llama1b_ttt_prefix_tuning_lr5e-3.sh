# python make_sbatch.py --time 48 --bash_files bash_commands/1203_llama1b_prefix_tuning_lr5e-3/1203_0_llama1b_ttt_prefix_tuning_lr5e-3.sh
# train ttt with raw meta llama1b, can ablate lora_to_output

# ttt prefix tuning with llama1b ntoken25 float32
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1203_0_llama1b_ttt_prefix_tuning_lr5e-3_float32_ntoken25 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 4 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format

# ttt prefix tuning with llama1b ntoken25 float16
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1203_0_llama1b_ttt_prefix_tuning_lr5e-3_float16_ntoken25 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 4 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --float16

# ttt prefix tuning with llama1b ntoken25 float32 flashattn
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1203_0_llama1b_ttt_prefix_tuning_lr5e-3_float32_flashattn_ntoken25 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 4 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --flash_attn

# ttt prefix tuning with llama1b ntoken25 float16 flashattn
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1203_0_llama1b_ttt_prefix_tuning_lr5e-3_float16_flashattn_ntoken25 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 4 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --flash_attn \
    --float16

# Submitted batch job 54227195
# Submitted batch job 54227196
# Submitted batch job 54227197
# Submitted batch job 54227198