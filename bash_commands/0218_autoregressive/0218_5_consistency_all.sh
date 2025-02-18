# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0218_autoregressive/0218_5_consistency_all.sh

# ar consistency0.1all
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_consistency0.1all \
    --consistency_type all \
    --consistency_loss_lambda 0.1 \
    --wandb

# ar consistency0.5all
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_consistency0.5all \
    --consistency_type all \
    --consistency_loss_lambda 0.5 \
    --wandb

# ar consistency1.0all
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_consistency1.0all \
    --consistency_type all \
    --consistency_loss_lambda 1.0 \
    --wandb

# Submitted batch job 57365005
# Submitted batch job 57365006
# Submitted batch job 57365007