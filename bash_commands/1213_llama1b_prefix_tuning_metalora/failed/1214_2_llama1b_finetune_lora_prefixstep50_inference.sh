# python make_sbatch.py --time 48 --bash_files bash_commands/1213_llama1b_prefix_tuning_metalora/1214_2_llama1b_finetune_lora_prefixstep50_inference.sh

# inference prefix tuning 50prefixstep loraepoch1
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1214_2_llama1b_finetune_lora_prefixstep50_inference_loraepoch1 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1214_1_llama1b_finetune_lora_prefixstep50_ttt_loraepoch1 \
    --lora_ckpt train_outputs/1214_0_llama1b_finetune_lora_two_stage_50prefixstep/checkpoint-outer-epoch1.pt \
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

# inference prefix tuning 50prefixstep loraepoch3
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1214_2_llama1b_finetune_lora_prefixstep50_inference_loraepoch3 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1214_1_llama1b_finetune_lora_prefixstep50_ttt_loraepoch3 \
    --lora_ckpt train_outputs/1214_0_llama1b_finetune_lora_two_stage_50prefixstep/checkpoint-outer-epoch3.pt \
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

# inference prefix tuning 50prefixstep loraepoch6
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1214_2_llama1b_finetune_lora_prefixstep50_inference_loraepoch6 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1214_1_llama1b_finetune_lora_prefixstep50_ttt_loraepoch6 \
    --lora_ckpt train_outputs/1214_0_llama1b_finetune_lora_two_stage_50prefixstep/checkpoint-outer-epoch6.pt \
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

# Submitted batch job 55017058 # terminated
# Submitted batch job 55017059 # terminated
# Submitted batch job 55017060 # terminated