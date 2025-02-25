# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0218_autoregressive/0218_21_weirdcast.sh

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0224/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0218_ar_weirdcast \
    --weird_cast \
    --wandb
