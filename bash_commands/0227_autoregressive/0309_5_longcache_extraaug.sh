# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0227_autoregressive/0309_5_longcache_extraaug.sh

# arlongcache extraaug0.3single
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0309_arlongcache_extraaug0.3single \
    --extra_augment_ratio 0.3 \
    --extra_augment_single_grid \
    --wandb

# arlongcache extraaug0.3nosingle
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0309_arlongcache_extraaug0.3nosingle \
    --extra_augment_ratio 0.3 \
    --extra_augment_single_grid \
    --wandb

# Submitted batch job 58107181
# Submitted batch job 58107182