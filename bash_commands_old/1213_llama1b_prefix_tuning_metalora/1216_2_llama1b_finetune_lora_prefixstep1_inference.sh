# python make_sbatch.py --time 48 --bash_files bash_commands/1213_llama1b_prefix_tuning_metalora/1216_2_llama1b_finetune_lora_prefixstep1_inference.sh

# inference prefix tuning 1prefixstep loraepoch5
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1216_2_llama1b_finetune_lora_prefixstep1_inference_loraepoch5 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1216_1_llama1b_finetune_lora_prefixstep1_ttt_loraepoch5 \
    --lora_ckpt train_outputs/1216_0_llama1b_finetune_lora_two_stage_1prefixstep/checkpoint-outer-epoch5.pt \
    --num_virtual_tokens 1 \
    --pt_epoch 20 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --limit_tokens

# inference prefix tuning 1prefixstep loraepoch10
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1216_2_llama1b_finetune_lora_prefixstep1_inference_loraepoch10 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1216_1_llama1b_finetune_lora_prefixstep1_ttt_loraepoch10 \
    --lora_ckpt train_outputs/1216_0_llama1b_finetune_lora_two_stage_1prefixstep/checkpoint-outer-epoch10.pt \
    --num_virtual_tokens 1 \
    --pt_epoch 20 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --limit_tokens

# inference prefix tuning 1prefixstep loraepoch25
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1216_2_llama1b_finetune_lora_prefixstep1_inference_loraepoch25 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1216_1_llama1b_finetune_lora_prefixstep1_ttt_loraepoch25 \
    --lora_ckpt train_outputs/1216_0_llama1b_finetune_lora_two_stage_1prefixstep/checkpoint-outer-epoch25.pt \
    --num_virtual_tokens 1 \
    --pt_epoch 20 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --limit_tokens

# inference prefix tuning 1prefixstep loraepoch50
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1216_2_llama1b_finetune_lora_prefixstep1_inference_loraepoch50 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1216_1_llama1b_finetune_lora_prefixstep1_ttt_loraepoch50 \
    --lora_ckpt train_outputs/1216_0_llama1b_finetune_lora_two_stage_1prefixstep/checkpoint-outer-epoch50.pt \
    --num_virtual_tokens 1 \
    --pt_epoch 20 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn \
    --limit_tokens

# Submitted batch job 55145541
# Submitted batch job 55145542
# Submitted batch job 55145543
# Submitted batch job 55145544