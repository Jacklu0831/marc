# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0317_noprogram/0327_0_ntokens.sh

# noprogram ntokens16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --no_bos \
    --tag 0317_noprogram_ntoken16 \
    --ntokens 16 \
    --eval_batch_size 8 \
    --wandb

# noprogram ntokens16 cutoff
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --no_bos \
    --tag 0317_noprogram_ntoken16_cutoff \
    --ntokens 16 \
    --attention_cutoff \
    --no_flash_attn \
    --eval_batch_size 8 \
    --wandb

# noprogram ntokens64 cutoff
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --no_bos \
    --tag 0317_noprogram_ntoken64_cutoff \
    --ntokens 64 \
    --attention_cutoff \
    --no_flash_attn \
    --eval_batch_size 8 \
    --wandb

# noprogram ntokens16 cutoff attendprev
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --no_bos \
    --tag 0317_noprogram_ntoken16_cutoff_attendprev \
    --ntokens 16 \
    --attention_cutoff \
    --attend_prev_programs \
    --no_flash_attn \
    --eval_batch_size 8 \
    --wandb

# noprogram ntokens64 cutoff attendprev
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --no_bos \
    --tag 0317_noprogram_ntoken64_cutoff_attendprev \
    --ntokens 64 \
    --attention_cutoff \
    --attend_prev_programs \
    --no_flash_attn \
    --eval_batch_size 8 \
    --wandb

# Submitted batch job 58761463
# Submitted batch job 58761464
# Submitted batch job 58761465
# Submitted batch job 58761467
# Submitted batch job 58761468