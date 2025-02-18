accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217_temp/train.py \
    --lr_scheduler constant \
    --warmup_epoch 1 \
    --tag test \
    --train_batch_size 2 \
    --grad_accum_steps 8 \
    --trainable_nbit 32


accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_truncated/train.py \
    --lr_scheduler constant \
    --warmup_epoch 1 \
    --tag test \
    --train_batch_size 2 \
    --grad_accum_steps 8 \
    --truncation_steps 100 \
    --trainable_nbit 32


accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_truncated/train.py \
    --lr_scheduler constant \
    --warmup_epoch 1 \
    --tag test \
    --train_batch_size 8 \
    --grad_accum_steps 2 \
    --truncation_steps 2 \
    --samples_per_epoch 64 \
    --debug_len 1867
