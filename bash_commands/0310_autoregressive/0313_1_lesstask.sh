# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0310_autoregressive/0313_1_lesstask.sh

# arlongcache 100task
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0313/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0313_arlongcache_100task \
    --train_data_dir ./data/re-arc/train_data_100/tasks \
    --wandb

# arlongcache 200task
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0313/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0313_arlongcache_200task \
    --train_data_dir ./data/re-arc/train_data_200/tasks \
    --wandb

# Submitted batch job 58265102
# Submitted batch job 58265103