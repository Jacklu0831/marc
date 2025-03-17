# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0317_autoregressive/0317_3_consistencyloss.sh

# arlongcache consistencyloss0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_consistencyloss0.1 \
    --consistency_loss_lambda 0.1 \
    --wandb

# arlongcache consistencyloss0.3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_consistencyloss0.3 \
    --consistency_loss_lambda 0.3 \
    --wandb

# arlongcache consistencyloss1.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_consistencyloss1.0 \
    --consistency_loss_lambda 1.0 \
    --wandb
