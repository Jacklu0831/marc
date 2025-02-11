# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0209_autoregressive/0209_1_ntokens.sh

# ar ntokens4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --tag 0209_ar_ntokens4 \
    --train_no_sample \
    --eval_no_sample \
    --ntokens 4 \
    --wandb

# ar ntokens8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --tag 0209_ar_ntokens8 \
    --train_no_sample \
    --eval_no_sample \
    --ntokens 8 \
    --wandb

# ar ntokens16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --tag 0209_ar_ntokens16 \
    --train_no_sample \
    --eval_no_sample \
    --ntokens 16 \
    --wandb

# ar ntokens32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --tag 0209_ar_ntokens32 \
    --train_no_sample \
    --eval_no_sample \
    --ntokens 32 \
    --wandb

# ar ntokens128
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --tag 0209_ar_ntokens128 \
    --train_no_sample \
    --eval_no_sample \
    --ntokens 128 \
    --wandb

# Submitted batch job 57077661
# Submitted batch job 57077662
# Submitted batch job 57077663
# Submitted batch job 57077664
# Submitted batch job 57077665 # cancelled loss jump