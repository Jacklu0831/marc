# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0213_autoregressive/0213_1_ntokens.sh

# ar ntokens8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_ntokens8 \
    --ntokens 8 \
    --wandb

# ar ntokens16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_ntokens16 \
    --ntokens 16 \
    --wandb

# ar ntokens64
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_ntokens64 \
    --ntokens 64 \
    --wandb

# ar ntokens128
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_ntokens128 \
    --ntokens 128 \
    --wandb

# Submitted batch job 57246035
# Submitted batch job 57246036
# Submitted batch job 57246037
# Submitted batch job 57246038