# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0218_autoregressive/0218_22_fsq.sh

# ar fsq1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0218_ar_fsq1024 \
    --no_normalize \
    --fsq_L 8 5 5 5 \
    --wandb

# ar fsq4096
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0218_ar_fsq4096 \
    --no_normalize \
    --fsq_L 7 5 5 5 5 \
    --wandb

# ar fsq16384
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0218_ar_fsq16384 \
    --no_normalize \
    --fsq_L 8 8 8 6 5 \
    --wandb
