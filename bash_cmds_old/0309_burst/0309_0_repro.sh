# python make_sbatch.py --ngpu 4 --time 48 --bash_files bash_commands/0309_burst/0309_0_repro.sh --burst

# arlongcache burst
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0309_arlongcache_burst \
    --train_batch_size 1 \
    --eval_batch_size 2 \
    --wandb

# noprogram burst
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag 0309_noprogram_burst \
    --train_batch_size 1 \
    --eval_batch_size 8 \
    --wandb

# Submitted batch job 35977
# Submitted batch job 35978