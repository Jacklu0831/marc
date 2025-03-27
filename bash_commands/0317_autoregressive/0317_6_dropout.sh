# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0317_autoregressive/0317_6_dropout.sh

# arlongcache fulldemonstrationdropout0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_fulldemonstrationdropout0.1 \
    --full_demonstration_dropout 0.1 \
    --wandb

# arlongcache fulldemonstrationdropout0.3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_fulldemonstrationdropout0.3 \
    --full_demonstration_dropout 0.3 \
    --wandb

# arlongcache partialdemonstrationdropout0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_partialdemonstrationdropout0.1 \
    --partial_demonstration_dropout 0.1 \
    --wandb

# arlongcache partialdemonstrationdropout0.3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_partialdemonstrationdropout0.3 \
    --partial_demonstration_dropout 0.3 \
    --wandb

# Submitted batch job 58466800
# Submitted batch job 58466801
# Submitted batch job 58466802
# Submitted batch job 58466803