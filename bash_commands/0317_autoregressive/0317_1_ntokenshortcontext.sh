# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0317_autoregressive/0317_1_ntokenshortcontext.sh

# arlongcache ntoken1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_ntoken1 \
    --ntokens 1 \
    --wandb

# arlongcache ntoken16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_ntoken16 \
    --ntokens 16 \
    --wandb

# arlongcache ntoken64
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_ntoken64 \
    --ntokens 64 \
    --wandb

# arlongcache shortcontext ntoken4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_shortcontext_ntoken4 \
    --short_context \
    --ntokens 4 \
    --wandb

# arlongcache shortcontext ntoken16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_shortcontext_ntoken16 \
    --short_context \
    --ntokens 16 \
    --wandb

# arlongcache shortcontext ntoken64
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_shortcontext_ntoken64 \
    --short_context \
    --ntokens 64 \
    --wandb

# Submitted batch job 58466828
# Submitted batch job 58466829
# Submitted batch job 58466830
# Submitted batch job 58466831
# Submitted batch job 58466832
# Submitted batch job 58466833

# resume arlongcache shortcontext ntoken4, 16, 64
# Submitted batch job 58648637
# Submitted batch job 58648638
# Submitted batch job 58648625

# resume arlongcache ntoken1, 16, 64
# Submitted batch job 58698161
# Submitted batch job 58698164
# Submitted batch job 58698165