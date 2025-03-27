# python make_sbatch.py --ngpu 2 --time 12 --gb 64 --bash_files bash_commands/0306_noprogram_nlp/0312_1_gpt2_burst.sh --burst

# noprogram nlp gpt2 nolora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0313_noprogram_nlp_gpt2_nolora_burst \
    --model_name gpt2 \
    --no_lora \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --wandb

# noprogram nlp llama1b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0313_noprogram_nlp_llama1b_burst \
    --model_name llama1b \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --wandb

# noprogram nlp llama1b excludefirst
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0313_noprogram_nlp_llama1b_excludefirst_burst \
    --model_name llama1b \
    --loss_type exclude_first \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --wandb

# noprogram nlp llama1b evalminpair2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0313_noprogram_nlp_llama1b_evalminpair2_burst \
    --model_name llama1b \
    --eval_min_num_pair 2 \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --wandb

# noprogram nlp llama1b excludefirst evalminpair2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0313_noprogram_nlp_llama1b_excludefirst_evalminpair2_burst \
    --model_name llama1b \
    --loss_type exclude_first \
    --eval_min_num_pair 2 \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --wandb

# noprogram nlp llama1b nolora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0313_noprogram_nlp_llama1b_nolora_burst \
    --model_name llama1b \
    --no_lora \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --wandb

# Submitted batch job 37410
# Submitted batch job 37411
# Submitted batch job 37412
# Submitted batch job 37413
# Submitted batch job 37414
# Submitted batch job 37415