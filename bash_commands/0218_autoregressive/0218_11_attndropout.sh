# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0218_autoregressive/0218_11_attndropout.sh

# ar attndropout0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0220/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_attndropout0.01 \
    --attention_dropout 0.01 \
    --wandb

# ar attndropout0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0220/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_attndropout0.03 \
    --attention_dropout 0.03 \
    --wandb

# ar attndropout0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0220/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_attndropout0.1 \
    --attention_dropout 0.1 \
    --wandb

# Submitted batch job 57468744
# Submitted batch job 57468745
# Submitted batch job 57468746