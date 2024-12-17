# python make_sbatch.py --time 48 --bash_files bash_commands/1213_llama1b_prefix_tuning_metalora/1214_2_llama1b_finetune_lora_prefixstep10_inference.sh

# inference prefix tuning 10prefixstep loraepoch1
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1214_2_llama1b_finetune_lora_prefixstep10_inference_loraepoch1 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1214_1_llama1b_finetune_lora_prefixstep10_ttt_loraepoch1 \
    --lora_ckpt train_outputs/1214_0_llama1b_finetune_lora_two_stage_10prefixstep/checkpoint-outer-epoch1.pt \
    --num_virtual_tokens 1 \
    --pt_epoch 20 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# inference prefix tuning 10prefixstep loraepoch3
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1214_2_llama1b_finetune_lora_prefixstep10_inference_loraepoch3 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1214_1_llama1b_finetune_lora_prefixstep10_ttt_loraepoch3 \
    --lora_ckpt train_outputs/1214_0_llama1b_finetune_lora_two_stage_10prefixstep/checkpoint-outer-epoch3.pt \
    --num_virtual_tokens 1 \
    --pt_epoch 20 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# inference prefix tuning 10prefixstep loraepoch6
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1214_2_llama1b_finetune_lora_prefixstep10_inference_loraepoch6 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1214_1_llama1b_finetune_lora_prefixstep10_ttt_loraepoch6 \
    --lora_ckpt train_outputs/1214_0_llama1b_finetune_lora_two_stage_10prefixstep/checkpoint-outer-epoch6.pt \
    --num_virtual_tokens 1 \
    --pt_epoch 20 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# inference prefix tuning 10prefixstep loraepoch9
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1214_2_llama1b_finetune_lora_prefixstep10_inference_loraepoch9 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1214_1_llama1b_finetune_lora_prefixstep10_ttt_loraepoch9 \
    --lora_ckpt train_outputs/1214_0_llama1b_finetune_lora_two_stage_10prefixstep/checkpoint-outer-epoch9.pt \
    --num_virtual_tokens 1 \
    --pt_epoch 20 \
    --temperature 0.0 \
    --n_sample 1 \
    --data_file kaggle_dataset/arc-agi_evaluation_challenges_selected.json \
    --solution_file kaggle_dataset/arc-agi_evaluation_solutions_selected.json \
    --include_n=1 \
    --new_format \
    --flash_attn

# Submitted batch job 55017054
# Submitted batch job 55017055
# Submitted batch job 55017056
# Submitted batch job 55017057