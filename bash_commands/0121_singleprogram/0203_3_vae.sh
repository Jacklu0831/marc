# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0121_singleprogram/0203_3_vae.sh

# single vae full lambda1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram/train.py \
    --tag 0203_single_vae_full_lambda1e-4 \
    --vae \
    --projection_type full \
    --kl_loss_lambda 1e-4 \
    --wandb

# single vae full lambda1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram/train.py \
    --tag 0203_single_vae_full_lambda1e-3 \
    --vae \
    --projection_type full \
    --kl_loss_lambda 1e-3 \
    --wandb

# single vae full lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram/train.py \
    --tag 0203_single_vae_full_lambda1e-2 \
    --vae \
    --projection_type full \
    --kl_loss_lambda 1e-2 \
    --wandb

# single vae full lambda1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram/train.py \
    --tag 0203_single_vae_full_lambda1e-1 \
    --vae \
    --projection_type full \
    --kl_loss_lambda 1e-1 \
    --wandb

# Submitted batch job 56865520
# Submitted batch job 56865521
# Submitted batch job 56865522
# Submitted batch job 56865523