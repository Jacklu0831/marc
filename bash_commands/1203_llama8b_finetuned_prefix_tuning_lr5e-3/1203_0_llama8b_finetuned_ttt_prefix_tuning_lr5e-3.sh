# python make_sbatch.py --time 48 --bash_files bash_commands/1203_llama8b_finetuned_prefix_tuning_lr5e-3/1203_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3.sh

# ttt prefix tuning with llama8b ntoken1
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1203_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_ntoken1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 1 \
    --grad_accum 2 \
    --epochs 10 \
    --num_virtual_tokens 1 \
    --learning_rate 5e-3 \
    --new_format \
    --flash_attn \
    --float16

# ttt prefix tuning with llama8b ntoken3
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1203_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_ntoken3 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 1 \
    --grad_accum 2 \
    --epochs 10 \
    --num_virtual_tokens 3 \
    --learning_rate 5e-3 \
    --new_format \
    --flash_attn \
    --float16

# ttt prefix tuning with llama8b ntoken5
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1203_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_ntoken5 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 1 \
    --grad_accum 2 \
    --epochs 10 \
    --num_virtual_tokens 5 \
    --learning_rate 5e-3 \
    --new_format \
    --flash_attn \
    --float16

# ttt prefix tuning with llama8b ntoken10
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1203_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_ntoken10 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 1 \
    --grad_accum 2 \
    --epochs 10 \
    --num_virtual_tokens 10 \
    --learning_rate 5e-3 \
    --new_format \
    --flash_attn \
    --float16

# ttt prefix tuning with llama8b ntoken25
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1203_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_ntoken25 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 1 \
    --grad_accum 2 \
    --epochs 10 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --flash_attn \
    --float16

# ttt prefix tuning with llama8b ntoken50
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1203_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_ntoken50 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 1 \
    --grad_accum 2 \
    --epochs 10 \
    --num_virtual_tokens 50 \
    --learning_rate 5e-3 \
    --new_format \
    --flash_attn \
    --float16

# OOM
# Submitted batch job 54178737
# Submitted batch job 54178738
# Submitted batch job 54178739
# Submitted batch job 54178740
# Submitted batch job 54178741
# Submitted batch job 54178742

# now with --float16 # killed due to debugging
# Submitted batch job 54183742
# Submitted batch job 54183743
# Submitted batch job 54183744
# Submitted batch job 54183745
# Submitted batch job 54183746
# Submitted batch job 54183747

# fix copy_()
# Submitted batch job 54227264
# Submitted batch job 54227265
# Submitted batch job 54227266
# Submitted batch job 54227267
# Submitted batch job 54227268
# Submitted batch job 54227269