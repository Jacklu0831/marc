# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0317_autoregressive/0323_0_vae.sh

# arlongcache kl1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --tag 0317_arlongcache_kl1e-4 \
    --vae \
    --kl_loss_lambda 1e-4 \
    --wandb

# arlongcache kl1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --tag 0317_arlongcache_kl1e-5 \
    --vae \
    --kl_loss_lambda 1e-5 \
    --wandb

# arlongcache kl1e-6
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --tag 0317_arlongcache_kl1e-6 \
    --vae \
    --kl_loss_lambda 1e-6 \
    --wandb

# # arlongcache kl1e-4 shortcontext
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
#     --lr_scheduler constant \
#     --tag 0317_arlongcache_kl1e-4_shortcontext \
#     --vae \
#     --kl_loss_lambda 1e-4 \
#     --short_context \
#     --wandb

# # arlongcache kl1e-5 shortcontext
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
#     --lr_scheduler constant \
#     --tag 0317_arlongcache_kl1e-5_shortcontext \
#     --vae \
#     --kl_loss_lambda 1e-5 \
#     --short_context \
#     --wandb

# # arlongcache kl1e-6 shortcontext
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
#     --lr_scheduler constant \
#     --tag 0317_arlongcache_kl1e-6_shortcontext \
#     --vae \
#     --kl_loss_lambda 1e-6 \
#     --short_context \
#     --wandb


# Submitted batch job 58758622
# Submitted batch job 58849248
# Submitted batch job 58849249
# didnt run shortcontext