# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0218_autoregressive/0218_12_programdropout.sh

# ar programdropout0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0220/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_programdropout0.01 \
    --program_dropout 0.01 \
    --wandb

# ar programdropout0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0220/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_programdropout0.03 \
    --program_dropout 0.03 \
    --wandb

# ar programdropout0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0220/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_programdropout0.1 \
    --program_dropout 0.1 \
    --wandb

# Submitted batch job 57468749
# Submitted batch job 57468750
# Submitted batch job 57468751