# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0211_autoregressive/0211_7_norm.sh

# ar norm
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0212/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_ar_norm \
    --normalize \
    --wandb

# Submitted batch job 57180929