# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0227_autoregressive/0227_14_longcache_consistencyloss.sh
# USES NTOKEN4
# with mixedprecisionfix

# arlongcache consistencyloss0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0305/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_consistencyloss0.1 \
    --consistency_loss_lambda 0.1 \
    --consistency_loss_offset_epochs 4 \
    --consistency_loss_linear_epochs 4

# arlongcache consistencyloss0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0305/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_consistencyloss0.5 \
    --consistency_loss_lambda 0.5 \
    --consistency_loss_offset_epochs 4 \
    --consistency_loss_linear_epochs 4

# arlongcache consistencyloss1.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0305/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_consistencyloss1.0 \
    --consistency_loss_lambda 1.0 \
    --consistency_loss_offset_epochs 4 \
    --consistency_loss_linear_epochs 4

# Submitted batch job 57967644
# Submitted batch job 57967645
# Submitted batch job 57967646