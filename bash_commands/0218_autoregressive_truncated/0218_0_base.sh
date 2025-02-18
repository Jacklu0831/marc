# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0218_autoregressive_truncated/0218_0_base.sh

# truncate2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_truncated/train.py \
    --lr_scheduler constant \
    --tag 0217_ar_truncate2 \
    --train_batch_size 8 \
    --grad_accum_steps 2 \
    --truncation_steps 2 \
    --wandb

# truncate4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_truncated/train.py \
    --lr_scheduler constant \
    --tag 0217_ar_truncate4 \
    --train_batch_size 4 \
    --grad_accum_steps 4 \
    --truncation_steps 4 \
    --wandb

# truncate100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_truncated/train.py \
    --lr_scheduler constant \
    --tag 0217_ar_truncate100 \
    --train_batch_size 2 \
    --grad_accum_steps 8 \
    --truncation_steps 100 \
    --wandb
