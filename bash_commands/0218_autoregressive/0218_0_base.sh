# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0218_autoregressive/0218_0_base.sh

# ar base
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_base \
    --wandb

# ar evalbatchsize2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_evalbatchsize2 \
    --eval_batch_size 2 \
    --wandb

# ar ntoken8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_ntoken8 \
    --ntokens 8 \
    --wandb

# ar ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_ntoken32 \
    --ntokens 32 \
    --wandb

# ar weightdecay0.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_weightdecay0.0 \
    --weight_decay 0.0 \
    --wandb

# ar noseparatecolor
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_noseparatecolor \
    --no_separate_color_tokens \
    --wandb

# Submitted batch job 57365018
# Submitted batch job 57365019
# Submitted batch job 57365020
# Submitted batch job 57365021
# Submitted batch job 57365022
# Submitted batch job 57365023