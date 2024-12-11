# python make_sbatch.py --time 48 --bash_files bash_commands/1209_llama1b_prefix_tuning_ntoken_lr_epoch_gridsearch/1129_0_ttt_search_lr1e-2.sh

# ttt search lr1e-2 ntoken1
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1129_0_ttt_search_lr1e-2_ntoken1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 1 \
    --learning_rate 1e-2 \
    --new_format \
    --flash_attn

# ttt search lr1e-2 ntoken3
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1129_0_ttt_search_lr1e-2_ntoken3 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 3 \
    --learning_rate 1e-2 \
    --new_format \
    --flash_attn

# ttt search lr1e-2 ntoken5
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1129_0_ttt_search_lr1e-2_ntoken5 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 5 \
    --learning_rate 1e-2 \
    --new_format \
    --flash_attn

# ttt search lr1e-2 ntoken10
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1129_0_ttt_search_lr1e-2_ntoken10 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 10 \
    --learning_rate 1e-2 \
    --new_format \
    --flash_attn

# ttt search lr1e-2 ntoken25
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1129_0_ttt_search_lr1e-2_ntoken25 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 25 \
    --learning_rate 1e-2 \
    --new_format \
    --flash_attn

# ttt search lr1e-2 ntoken50
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1129_0_ttt_search_lr1e-2_ntoken50 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 10 \
    --num_virtual_tokens 50 \
    --learning_rate 1e-2 \
    --new_format \
    --flash_attn

# Submitted batch job 54612056
# Submitted batch job 54612057
# Submitted batch job 54612058
# Submitted batch job 54612059
# Submitted batch job 54612060
# Submitted batch job 54612061
# trainlosses: 0.246, 0.336, 0.342, 0.263, 0.333, 0.362