# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0317_autoregressive/0317_7_shortcontext_lesstask.sh

# arlongcache shortcontext 100task
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_shortcontext_100task \
    --train_data_dir ./data/re-arc/train_data_100/tasks \
    --short_context \
    --wandb

# arlongcache shortcontext 200task
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_shortcontext_200task \
    --train_data_dir ./data/re-arc/train_data_200/tasks \
    --short_context \
    --wandb

# Submitted batch job 58630694
# Submitted batch job 58630695
