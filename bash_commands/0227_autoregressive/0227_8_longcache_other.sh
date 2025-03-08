# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0227_autoregressive/0227_8_longcache_other.sh
# 40 epochs in ~60 hours on 2gpu, ~80 hours notf32
# USES NTOKEN4

# arlongcache lossonfirst
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_lossonfirst \
    --loss_on_first \
    --wandb

# arlongcache nomaxseqlen
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_nomaxseqlen \
    --max_seq_len 10000000 \
    --wandb

# arlongcache notokenweighting
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --tag 0227_arlongcache_notokenweighting \
    --wandb

# arlongcache noresidual
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --tag 0227_arlongcache_noresidual \
    --no_residual \
    --wandb

# Submitted batch job 57857691
# Submitted batch job 57857692
# Submitted batch job 57857693
# Submitted batch job 57858174