# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0211_autoregressive/0211_10_consistency_type.sh

# ar consistency0.3onlyfirst
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0212/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0212_ar_consistency0.3onlyfirst \
    --consistency_type only_first \
    --consistency_loss_lambda 0.3 \
    --wandb

# ar consistency0.3includelast
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0212/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0212_ar_consistency0.3includelast \
    --consistency_type include_last \
    --consistency_loss_lambda 0.3 \
    --wandb

# Submitted batch job 57180935
# Submitted batch job 57180936