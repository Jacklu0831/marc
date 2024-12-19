# python make_sbatch.py --time 48 --bash_files bash_commands/1213_llama1b_prefix_tuning_metalora/1216_2_llama1b_finetune_lora_prefixstep50_inference.sh

# inference prefix tuning 50prefixstep loraepoch5
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1216_2_llama1b_finetune_lora_prefixstep50_inference_loraepoch5 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1216_1_llama1b_finetune_lora_prefixstep50_ttt_loraepoch5 \
    --lora_ckpt train_outputs/1216_0_llama1b_finetune_lora_two_stage_50prefixstep/checkpoint-outer-epoch5.pt \
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

# inference prefix tuning 50prefixstep loraepoch10
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1216_2_llama1b_finetune_lora_prefixstep50_inference_loraepoch10 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1216_1_llama1b_finetune_lora_prefixstep50_ttt_loraepoch10 \
    --lora_ckpt train_outputs/1216_0_llama1b_finetune_lora_two_stage_50prefixstep/checkpoint-outer-epoch10.pt \
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

# inference prefix tuning 50prefixstep loraepoch15
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1216_2_llama1b_finetune_lora_prefixstep50_inference_loraepoch15 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1216_1_llama1b_finetune_lora_prefixstep50_ttt_loraepoch15 \
    --lora_ckpt train_outputs/1216_0_llama1b_finetune_lora_two_stage_50prefixstep/checkpoint-outer-epoch15.pt \
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

# inference prefix tuning 50prefixstep loraepoch20
python predict_prefix_tuning.py \
    --experiment_folder inference_outputs/experiments/1216_2_llama1b_finetune_lora_prefixstep50_inference_loraepoch20 \
    --pretrained_checkpoint downloaded_models/meta-llama/Llama-3.2-1B-Instruct \
    --pt_checkpoints_folder train_outputs/1216_1_llama1b_finetune_lora_prefixstep50_ttt_loraepoch20 \
    --lora_ckpt train_outputs/1216_0_llama1b_finetune_lora_two_stage_50prefixstep/checkpoint-outer-epoch20.pt \
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
