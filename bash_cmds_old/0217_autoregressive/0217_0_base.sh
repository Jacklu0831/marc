# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0217_autoregressive/0217_0_base.sh

# ar base
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217_eval/train.py \
    --lr_scheduler constant \
    --tag 0217_ar_base \
    --wandb

# ar evalbatchsize2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217_eval/train.py \
    --lr_scheduler constant \
    --tag 0217_ar_evalbatchsize2 \
    --eval_batch_size 2 \
    --wandb

# ar ntoken16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217_eval/train.py \
    --lr_scheduler constant \
    --tag 0217_ar_ntoken16 \
    --ntokens 16 \
    --wandb

# ar ntoken64
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217_eval/train.py \
    --lr_scheduler constant \
    --tag 0217_ar_ntoken64 \
    --ntokens 64 \
    --wandb

# Submitted batch job 57338602
# Submitted batch job 57338603
# Submitted batch job 57338604
# Submitted batch job 57338605