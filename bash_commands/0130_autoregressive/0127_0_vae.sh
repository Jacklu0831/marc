# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0130_autoregressive/0127_0_vae.sh

# ar kl1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive/train.py \
    --tag 0202_ar_kl1e-3 \
    --kl_loss_lambda 1e-3 \
    --wandb

# ar kl1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive/train.py \
    --tag 0202_ar_kl1e-4 \
    --kl_loss_lambda 1e-4 \
    --wandb

# ar kl1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive/train.py \
    --tag 0202_ar_kl1e-5 \
    --kl_loss_lambda 1e-5 \
    --wandb

# ar kl1e-6
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive/train.py \
    --tag 0202_ar_kl1e-6 \
    --kl_loss_lambda 1e-6 \
    --wandb

# Submitted batch job 56923706
# Submitted batch job 56923707
# Submitted batch job 56923708
# Submitted batch job 56923709