# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0227_autoregressive/0227_3_fsq.sh

# ar fsq1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arshort_fsq1024 \
    --no_normalize \
    --fsq_L 8 5 5 5 \
    --wandb

# ar fsq4096
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arshort_fsq4096 \
    --no_normalize \
    --fsq_L 7 5 5 5 5 \
    --wandb

# ar fsq16384
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arshort_fsq16384 \
    --no_normalize \
    --fsq_L 8 8 8 6 5 \
    --wandb

# Submitted batch job 57706030
# Submitted batch job 57706031
# Submitted batch job 57706032
# terrible performance, axed