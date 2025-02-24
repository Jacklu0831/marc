# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0218_autoregressive/0218_0_highweightdecay.sh

# ar weightdecay0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_weightdecay0.03 \
    --weight_decay 0.03 \
    --wandb

# ar weightdecay0.05
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_weightdecay0.05 \
    --weight_decay 0.05 \
    --wandb

# Submitted batch job 57410308
# Submitted batch job 57410309