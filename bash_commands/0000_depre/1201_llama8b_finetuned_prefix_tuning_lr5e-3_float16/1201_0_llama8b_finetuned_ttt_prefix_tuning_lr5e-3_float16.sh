# python make_sbatch.py --time 48 --bash_files bash_commands/1201_llama8b_finetuned_prefix_tuning_lr5e-3_float16/1201_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_float16.sh

# ttt prefix tuning with llama8b ntoken1 float16
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1129_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_float16_ntoken1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 1 \
    --learning_rate 5e-3 \
    --new_format \
    --float16

# ttt prefix tuning with llama8b ntoken3 float16
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1129_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_float16_ntoken3 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 3 \
    --learning_rate 5e-3 \
    --new_format \
    --float16

# ttt prefix tuning with llama8b ntoken5 float16
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1129_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_float16_ntoken5 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 5 \
    --learning_rate 5e-3 \
    --new_format \
    --float16

# ttt prefix tuning with llama1b ntoken10 float16
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1129_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_float16_ntoken10 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 10 \
    --learning_rate 5e-3 \
    --new_format \
    --float16

# ttt prefix tuning with llama1b ntoken25 float16
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1129_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_float16_ntoken25 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --float16

# ttt prefix tuning with llama1b ntoken50 float16
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1129_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_float16_ntoken50 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 50 \
    --learning_rate 5e-3 \
    --new_format \
    --float16

# Submitted batch job 54134022
# Submitted batch job 54134023
# Submitted batch job 54134024
# Submitted batch job 54134025
# Submitted batch job 54134026
# Submitted batch job 54134027