# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0211_autoregressive/0211_9_consistency_loss.sh

# ar consistency0.1all
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0212/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0212_ar_consistency0.1all \
    --consistency_type all \
    --consistency_loss_lambda 0.1 \
    --wandb

# ar consistency0.3all
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0212/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0212_ar_consistency0.3all \
    --consistency_type all \
    --consistency_loss_lambda 0.3 \
    --wandb

# ar consistency1.0all
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0212/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0212_ar_consistency1.0all \
    --consistency_type all \
    --consistency_loss_lambda 1.0 \
    --wandb

# Submitted batch job 57180932
# Submitted batch job 57180933
# Submitted batch job 57180934