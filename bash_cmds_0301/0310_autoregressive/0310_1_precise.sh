# python make_sbatch.py --ngpu 4 --time 48 --bash_files bash_commands/0310_autoregressive/0310_1_precise.sh

# arlongcache precise nobos
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0310_arlongcache_precise_nobos \
    --no_bos \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --no_flash_attn \
    --train_batch_size 1 \
    --eval_batch_size 2 \
    --wandb

# Submitted batch job 58160591