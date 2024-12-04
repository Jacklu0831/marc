# python make_sbatch.py --time 48 --bash_files bash_commands/1203_llama8b_finetuned_prefix_tuning_lr5e-3_nogradaccum/1203_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_nogradaccum.sh

# ttt prefix tuning with llama8b nogradaccum ntoken1
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1203_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_nogradaccum_ntoken1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 1 \
    --learning_rate 5e-3 \
    --new_format \
    --flash_attn \
    --float16

# ttt prefix tuning with llama8b nogradaccum ntoken3
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1203_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_nogradaccum_ntoken3 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 3 \
    --learning_rate 5e-3 \
    --new_format \
    --flash_attn \
    --float16

# ttt prefix tuning with llama8b nogradaccum ntoken5
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1203_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_nogradaccum_ntoken5 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 5 \
    --learning_rate 5e-3 \
    --new_format \
    --flash_attn \
    --float16

# ttt prefix tuning with llama8b nogradaccum ntoken10
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1203_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_nogradaccum_ntoken10 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 10 \
    --learning_rate 5e-3 \
    --new_format \
    --flash_attn \
    --float16

# ttt prefix tuning with llama8b nogradaccum ntoken25
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1203_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_nogradaccum_ntoken25 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --flash_attn \
    --float16

# ttt prefix tuning with llama8b nogradaccum ntoken50
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/ekinakyurek/marc-8B-finetuned-llama3 \
    --experiment_folder train_outputs/1203_0_llama8b_finetuned_ttt_prefix_tuning_lr5e-3_nogradaccum_ntoken50 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 50 \
    --learning_rate 5e-3 \
    --new_format \
    --flash_attn \
    --float16

# Submitted batch job 54244971
# Submitted batch job 54244972
# Submitted batch job 54244973
# Submitted batch job 54244974
# Submitted batch job 54244975
# Submitted batch job 54244976