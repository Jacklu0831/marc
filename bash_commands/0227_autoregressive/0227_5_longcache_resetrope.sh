# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0227_autoregressive/0227_5_longcache_resetrope.sh

# arlongcache demondropout0.03 resetrope
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0303/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_demondropout0.03_resetrope \
    --demonstration_dropout 0.03 \
    --reset_rope \
    --wandb

# arlongcache demondropout0.1 resetrope
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0303/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_demondropout0.1_resetrope \
    --demonstration_dropout 0.1 \
    --reset_rope \
    --wandb

# arlongcache demondropout0.3 resetrope
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0303/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_demondropout0.3_resetrope \
    --demonstration_dropout 0.3 \
    --reset_rope \
    --wandb

# Submitted batch job 57840724
# Submitted batch job 57840725
# Submitted batch job 57840727