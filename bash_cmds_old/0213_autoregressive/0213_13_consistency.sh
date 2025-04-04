# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0213_autoregressive/0213_13_consistency.sh

# ar consistency0.1all
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_consistency0.1all \
    --consistency_type all \
    --consistency_loss_lambda 0.1 \
    --wandb

# ar consistency0.3all
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_consistency0.3all \
    --consistency_type all \
    --consistency_loss_lambda 0.3 \
    --wandb

# ar consistency1.0all
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_consistency1.0all \
    --consistency_type all \
    --consistency_loss_lambda 1.0 \
    --wandb

# ar consistency0.3onlyfirst
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_consistency0.3onlyfirst \
    --consistency_type only_first \
    --consistency_loss_lambda 0.3 \
    --wandb

# ar consistency0.3excludelast
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_consistency0.3excludelast \
    --consistency_type exclude_last \
    --consistency_loss_lambda 0.3 \
    --wandb

# Submitted batch job 57329462
# Submitted batch job 57329463
# Submitted batch job 57329464
# Submitted batch job 57329465
# Submitted batch job 57329466