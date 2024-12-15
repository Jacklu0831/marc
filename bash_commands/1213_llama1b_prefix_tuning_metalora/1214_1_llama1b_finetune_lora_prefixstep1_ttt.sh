# python make_sbatch.py --time 48 --bash_files bash_commands/1213_llama1b_prefix_tuning_metalora/1214_1_llama1b_finetune_lora_prefixstep1_ttt.sh

# ttt prefix tuning 1prefixstep loraepoch1
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1214_1_llama1b_finetune_lora_prefixstep1_ttt_loraepoch1 \
    --lora_ckpt train_outputs/1214_0_llama1b_finetune_lora_two_stage_1prefixstep/checkpoint-outer-epoch1.pt \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 20 \
    --num_virtual_tokens 1 \
    --learning_rate 1e-1 \
    --new_format \
    --flash_attn

# ttt prefix tuning 1prefixstep loraepoch
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1214_1_llama1b_finetune_lora_prefixstep1_ttt_loraepoch1 \
    --lora_ckpt train_outputs/1214_0_llama1b_finetune_lora_two_stage_1prefixstep/checkpoint-outer-epoch1.pt \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 20 \
    --num_virtual_tokens 1 \
    --learning_rate 1e-1 \
    --new_format \
    --flash_attn

# add more epochs based on how many epochs are done... think about how to eval properly,
# need to be able to compare the same lora epoch at differet number of prefix steps
