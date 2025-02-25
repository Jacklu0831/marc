# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0218_autoregressive/0218_14_vae.sh

# ar vae kl1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0221/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_vae_kl1e-5 \
    --vae \
    --kl_loss_lambda 1e-5 \
    --wandb

# ar vae kl1e-6
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0221/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_vae_kl1e-6 \
    --vae \
    --kl_loss_lambda 1e-6 \
    --wandb

# ar vae kl1e-7
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0221/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_vae_kl1e-7 \
    --vae \
    --kl_loss_lambda 1e-7 \
    --wandb

# Submitted batch job 57597350
# Submitted batch job 57597351
# Submitted batch job 57597352