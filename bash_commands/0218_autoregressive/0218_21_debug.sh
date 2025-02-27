# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0218_autoregressive/0218_21_weirdcast.sh

# ar weirdcast
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0218_ar_weirdcast \
    --weird_cast \
    --wandb

# ar float32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0218_ar_float32 \
    --untrainable_nbit 16 \
    --trainable_nbit 32 \
    --eval_batch_size 32 \
    --no_flash_attn \
    --wandb

# ar float32 notf32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0218_ar_float32_notf32 \
    --untrainable_nbit 16 \
    --trainable_nbit 32 \
    --eval_batch_size 32 \
    --no_flash_attn \
    --no_tf32 \
    --wandb

# ar trainbs1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0218_ar_trainbs1 \
    --train_batch_size 1 \
    --grad_accum_steps 16 \
    --wandb
