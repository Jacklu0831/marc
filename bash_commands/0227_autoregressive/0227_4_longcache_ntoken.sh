# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0227_autoregressive/0227_4_longcache_ntoken.sh
# 40 epochs in ~60 hours on 2gpu, ~80 hours notf32

# arlongcache ntokens4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_ntokens4 \
    --ntokens 4 \
    --wandb

# arlongcache ntokens8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_ntokens8 \
    --ntokens 8 \
    --wandb

# arlongcache ntokens32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_ntokens32 \
    --ntokens 32 \
    --wandb

# arlongcache ntokens64
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_ntokens64 \
    --ntokens 64 \
    --wandb

# Submitted batch job 57839960
# Submitted batch job 57839961
# Submitted batch job 57839962
# Submitted batch job 57839963