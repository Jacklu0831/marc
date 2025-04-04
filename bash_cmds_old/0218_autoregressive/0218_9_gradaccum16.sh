# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0218_autoregressive/0218_9_gradaccum16.sh

# ar gradaccum16 lr4e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_gradaccum16_lr4e-4 \
    --grad_accum_steps 16 \
    --lr_embedding 4e-5 \
    --lr_program 4e-4 \
    --lr_prior 4e-4 \
    --lr_other 4e-4 \
    --wandb

# ar gradaccum16 lr6e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_gradaccum16_lr6e-4 \
    --grad_accum_steps 16 \
    --lr_embedding 6e-5 \
    --lr_program 6e-4 \
    --lr_prior 6e-4 \
    --lr_other 6e-4 \
    --wandb

# ar gradaccum16 lr8e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_gradaccum16_lr8e-4 \
    --grad_accum_steps 16 \
    --lr_embedding 8e-5 \
    --lr_program 8e-4 \
    --lr_prior 8e-4 \
    --lr_other 8e-4 \
    --wandb

# Submitted batch job 57467118
# Submitted batch job 57467119
# Submitted batch job 57467120