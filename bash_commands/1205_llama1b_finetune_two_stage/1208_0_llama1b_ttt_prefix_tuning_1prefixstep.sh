# python make_sbatch.py --time 48 --bash_files bash_commands/1205_llama1b_finetune_two_stage/1208_0_llama1b_ttt_prefix_tuning_1prefixstep.sh

# ttt prefix tuning 1prefixstep loraepoch1
python test_time_train_prefix_tuning.py \
    --tokenizer_path downloaded_models/meta-llama/Llama-3.2-1B-Instruct/original/tokenizer.model \
    --base_checkpoint_dir downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --experiment_folder train_outputs/1208_0_llama1b_ttt_prefix_tuning_1prefixstep_loraepoch1 \
    --lora_ckpt train_outputs/1205_0_llama1b_finetune_lora_two_stage_1prefixstep_1lorastep_reuse/checkpoint-outer-epoch1.pt \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --batch_size 2 \
    --epochs 2 \
    --num_virtual_tokens 25 \
    --learning_rate 5e-3 \
    --new_format \
    --flash_attn
