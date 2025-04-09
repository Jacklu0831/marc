# python make_sbatch.py --ngpu 1 --time 48 --bash_files bash_commands/0317_noprogram_nlp/0317_1_variable_num_pair.sh

# noprogram nlp llama1b lora excludefirst
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0317_noprogram_nlp_llama1b_lora_excludefirst \
    --model_name llama1b \
    --max_seq_len 8192 \
    --max_pair_len 2048 \
    --train_batch_size 2 \
    --grad_accum_steps 4 \
    --eval_batch_size 4 \
    --eval_pretrained \
    --loss_type exclude_first \
    --wandb

# noprogram nlp llama1b lora evalminpair2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0317_noprogram_nlp_llama1b_lora_evalminpair2 \
    --model_name llama1b \
    --max_seq_len 8192 \
    --max_pair_len 2048 \
    --train_batch_size 2 \
    --grad_accum_steps 4 \
    --eval_batch_size 4 \
    --eval_pretrained \
    --eval_min_num_pair 2 \
    --wandb

# noprogram nlp llama1b lora excludefirst evalminpair2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0317_noprogram_nlp_llama1b_lora_excludefirst_evalminpair2 \
    --model_name llama1b \
    --max_seq_len 8192 \
    --max_pair_len 2048 \
    --train_batch_size 2 \
    --grad_accum_steps 4 \
    --eval_batch_size 4 \
    --eval_pretrained \
    --loss_type exclude_first \
    --eval_min_num_pair 2 \
    --wandb

# Submitted batch job 58606611
# Submitted batch job 58630048
# Submitted batch job 58630049