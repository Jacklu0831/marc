# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0213_autoregressive/0213_14_noinfoleak.sh

# ar noinfoleak
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_noinfoleak \
    --wandb

# Submitted batch job 57329525