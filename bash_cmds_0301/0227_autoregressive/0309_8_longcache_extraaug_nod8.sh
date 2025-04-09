# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0227_autoregressive/0309_8_longcache_extraaug_nod8.sh

# arlongcache extraaug0.3single nod8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0309_arlongcache_extraaug0.3single_nod8 \
    --extra_augment_ratio 0.3 \
    --extra_augment_single_grid \
    --no_d8 \
    --wandb

# arlongcache extraaug0.3nosingle nod8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0309_arlongcache_extraaug0.3nosingle_nod8 \
    --extra_augment_ratio 0.3 \
    --extra_augment_single_grid \
    --no_d8 \
    --wandb

# arlongcache extraaug0.3single nod8 noothers
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0309_arlongcache_extraaug0.3single_nod8_noothers \
    --extra_augment_ratio 0.3 \
    --extra_augment_single_grid \
    --no_d8 \
    --no_color_permute \
    --no_pair_permute \
    --wandb

# arlongcache extraaug0.3nosingle nod8 noothers
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0309_arlongcache_extraaug0.3nosingle_nod8_noothers \
    --extra_augment_ratio 0.3 \
    --extra_augment_single_grid \
    --no_d8 \
    --no_color_permute \
    --no_pair_permute \
    --wandb

# Submitted batch job 58174537
# Submitted batch job 58174538
# Submitted batch job 58174539
# Submitted batch job 58174540