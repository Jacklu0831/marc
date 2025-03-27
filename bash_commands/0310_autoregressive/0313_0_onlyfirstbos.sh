# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0310_autoregressive/0313_0_onlyfirstbos.sh

# arlongcache onlyfirstbos
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0313/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0313_arlongcache_onlyfirstbos \
    --only_first_bos \
    --wandb

# Submitted batch job 58265101