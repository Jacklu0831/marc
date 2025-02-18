# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0213_autoregressive/0213_4_misc.sh

# ar nodim
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_nodim \
    --no_dim \
    --wandb

# ar evalbatchsize1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_evalbatchsize1 \
    --eval_batch_size 1 \
    --wandb

# ar evalbatchsize4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_evalbatchsize4 \
    --eval_batch_size 4 \
    --wandb

# Submitted batch job 57246071
# Submitted batch job 57246073
# Submitted batch job 57246074