# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0218_autoregressive/0218_6_consistency_excludelast.sh

# ar consistency0.1excludelast
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_consistency0.1excludelast \
    --consistency_type exclude_last \
    --consistency_loss_lambda 0.1 \
    --wandb

# ar consistency0.5excludelast
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_consistency0.5excludelast \
    --consistency_type exclude_last \
    --consistency_loss_lambda 0.5 \
    --wandb

# ar consistency1.0excludelast
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_consistency1.0excludelast \
    --consistency_type exclude_last \
    --consistency_loss_lambda 1.0 \
    --wandb

# Submitted batch job 57364507
# Submitted batch job 57364508
# Submitted batch job 57364509