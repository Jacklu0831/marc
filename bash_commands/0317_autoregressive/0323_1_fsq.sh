# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0317_autoregressive/0323_1_fsq.sh

# arlongcache fsq1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_fsq1024 \
    --no_normalize \
    --fsq_L 8 5 5 5 \
    --wandb

# arlongcache fsq4096
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_fsq4096 \
    --no_normalize \
    --fsq_L 7 5 5 5 5 \
    --wandb

# arlongcache fsq16384
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_fsq16384 \
    --no_normalize \
    --fsq_L 8 8 8 6 5 \
    --wandb

# arlongcache fsq1024 shortcontext
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_fsq1024_shortcontext \
    --no_normalize \
    --fsq_L 8 5 5 5 \
    --short_context \
    --wandb

# arlongcache fsq4096 shortcontext
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_fsq4096_shortcontext \
    --no_normalize \
    --fsq_L 7 5 5 5 5 \
    --short_context \
    --wandb

# arlongcache fsq16384 shortcontext
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_fsq16384_shortcontext \
    --no_normalize \
    --fsq_L 8 8 8 6 5 \
    --short_context \
    --wandb
