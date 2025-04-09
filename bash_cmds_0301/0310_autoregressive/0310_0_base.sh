# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0310_autoregressive/0310_0_base.sh

# arlongcache repro
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0310/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0310_arlongcache_repro \
    --wandb

# arlongcache nobos
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0310/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0310_arlongcache_nobos \
    --no_bos \
    --wandb

# arlongcache shortcontext nobos
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0310/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0310_arlongcache_shortcontext_nobos \
    --no_bos \
    --short_context \
    --wandb

# Submitted batch job 58139590
# Submitted batch job 58139591
# Submitted batch job 58139592