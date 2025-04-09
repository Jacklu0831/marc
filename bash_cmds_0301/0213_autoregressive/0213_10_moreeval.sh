# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0213_autoregressive/0213_10_moreeval.sh

# ar evallast
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0215/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_evallast \
    --eval_last_program \
    --wandb

# ar evallast_evalbatchsize64
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0215/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_evallast_evalbatchsize64 \
    --eval_last_program \
    --eval_batch_size 64 \
    --wandb

# ar evalbatchsize32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0215/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_evalbatchsize32 \
    --eval_batch_size 32 \
    --wandb

# ar evalbatchsize64
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0215/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_evalbatchsize64 \
    --eval_batch_size 64 \
    --wandb

# Submitted batch job 57319926
# Submitted batch job 57329344
# Submitted batch job 57319927
# Submitted batch job 57319928