# python make_sbatch.py --time 48 --bash_files bash_commands/1201_llama1b_prefix_tuning_lr5e-3_float16/1201_0_llama1b_ttt_prefix_tuning_lr5e-3_float16.sh
# train ttt with raw meta llama1b, can ablate lora_to_output

# ttt prefix tuning with llama1b ntoken1 float16
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1201_0_llama1b_ttt_prefix_tuning_lr5e-3_float16_ntoken1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 4 \
    --num_virtual_tokens 1 \
    --learning_rate 5e-3 \
    --new_format \
    --float16

# ttt prefix tuning with llama1b ntoken3 float16
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1201_0_llama1b_ttt_prefix_tuning_lr5e-3_float16_ntoken3 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 4 \
    --num_virtual_tokens 3 \
    --learning_rate 5e-3 \
    --new_format \
    --float16

# ttt prefix tuning with llama1b ntoken5 float16
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1201_0_llama1b_ttt_prefix_tuning_lr5e-3_float16_ntoken5 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 4 \
    --num_virtual_tokens 5 \
    --learning_rate 5e-3 \
    --new_format \
    --float16

# ttt prefix tuning with llama1b ntoken10 float16
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1201_0_llama1b_ttt_prefix_tuning_lr5e-3_float16_ntoken10 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 4 \
    --num_virtual_tokens 10 \
    --learning_rate 5e-3 \
    --new_format \
    --float16

# ttt prefix tuning with llama1b ntoken25 float16
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1201_0_llama1b_ttt_prefix_tuning_lr5e-3_float16_ntoken25 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 4 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --float16

# ttt prefix tuning with llama1b ntoken50 float16
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1201_0_llama1b_ttt_prefix_tuning_lr5e-3_float16_ntoken50 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 4 \
    --num_virtual_tokens 50 \
    --learning_rate 5e-3 \
    --new_format \
    --float16

# Submitted batch job 54134003
# Submitted batch job 54134004
# Submitted batch job 54134005
# Submitted batch job 54134006
# Submitted batch job 54134007
# Submitted batch job 54134008