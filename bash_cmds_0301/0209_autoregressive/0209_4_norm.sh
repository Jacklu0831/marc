# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0209_autoregressive/0209_4_norm.sh

# ar norm
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --tag 0211_ar_norm \
    --train_no_sample \
    --eval_no_sample \
    --normalize \
    --wandb

# ar residual norm
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --tag 0211_ar_residual_norm \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --normalize \
    --wandb
