# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0227_autoregressive/0227_13_longcache_newntoken.sh
# USES NTOKEN4
# with mixedprecisionfix

# arlongcache ntokens2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0305/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_ntokens2 \
    --ntokens 2 \
    --wandb

# arlongcache ntokens0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0305/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_ntokens0 \
    --ntokens 0 \
    --wandb

# Submitted batch job 57963103
# Submitted batch job 57963104