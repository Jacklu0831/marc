# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0218_autoregressive/0218_7_consistency_onlyfirst.sh

# ar consistency0.1onlyfirst
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_consistency0.1onlyfirst \
    --consistency_type only_first \
    --consistency_loss_lambda 0.1 \
    --wandb

# ar consistency0.5onlyfirst
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_consistency0.5onlyfirst \
    --consistency_type only_first \
    --consistency_loss_lambda 0.5 \
    --wandb

# ar consistency1.0onlyfirst
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_consistency1.0onlyfirst \
    --consistency_type only_first \
    --consistency_loss_lambda 1.0 \
    --wandb

# Submitted batch job 57364495
# Submitted batch job 57364496
# Submitted batch job 57364497