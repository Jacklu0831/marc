# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0218_autoregressive/0218_13_programnoise.sh

# ar programnoise0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0220/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_programnoise0.01 \
    --program_noise_std 0.01 \
    --wandb

# ar programnoise0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0220/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_programnoise0.03 \
    --program_noise_std 0.03 \
    --wandb

# ar programnoise0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0220/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_programnoise0.1 \
    --program_noise_std 0.1 \
    --wandb

# Submitted batch job 57468754
# Submitted batch job 57468755
# Submitted batch job 57468756