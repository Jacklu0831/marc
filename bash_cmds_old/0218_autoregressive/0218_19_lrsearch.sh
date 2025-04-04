# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0218_autoregressive/0218_19_lrsearch.sh

# ar lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0223/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0218_ar_lr1e-4 \
    --lr_embedding 1e-5 \
    --lr_program 1e-4 \
    --lr_prior 1e-4 \
    --lr_other 1e-4 \
    --wandb

# ar lr3e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0223/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0218_ar_lr3e-4 \
    --lr_embedding 3e-5 \
    --lr_program 3e-4 \
    --lr_prior 3e-4 \
    --lr_other 3e-4 \
    --wandb

# ar lr4e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0223/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0218_ar_lr4e-4 \
    --lr_embedding 4e-5 \
    --lr_program 4e-4 \
    --lr_prior 4e-4 \
    --lr_other 4e-4 \
    --wandb

# Submitted batch job 57577326
# Submitted batch job 57577327
# Submitted batch job 57577328