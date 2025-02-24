# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0218_autoregressive/0218_17_tokenweighted.sh

# ar tokenweighted
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0223/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_tokenweighted \
    --token_weighted \
    --wandb

# Submitted batch job 57540159