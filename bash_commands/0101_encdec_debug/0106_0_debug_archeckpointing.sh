accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0219/train.py \
    --lr_scheduler constant \
    --tag test \
    --train_batch_size 2 \
    --grad_accum_steps 8 \
    --num_epochs 40

# 100 hours, 52GB


accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0219/train.py \
    --lr_scheduler constant \
    --tag test \
    --train_batch_size 16 \
    --grad_accum_steps 1 \
    --num_epochs 40 \
    --ar_gradient_checkpointing

# 156 hours, 60GB




accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_truncated_0219/train.py \
    --lr_scheduler constant \
    --tag test \
    --train_batch_size 2 \
    --grad_accum_steps 4 \
    --num_epochs 40 \
    --debug_len 1867 \
    --truncation_steps 100

# 10 iters -> 179 hours, 60GB

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_truncated_0219/train.py \
    --lr_scheduler constant \
    --tag test \
    --train_batch_size 2 \
    --grad_accum_steps 4 \
    --num_epochs 40 \
    --debug_len 1867 \
    --truncation_steps 2

# 10 iters -> 180 hours, 18.7GB
