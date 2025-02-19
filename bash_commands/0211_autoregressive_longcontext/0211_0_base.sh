# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0211_autoregressive_longcontext/0211_0_base.sh

# arlong base
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_arlong_base \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --grad_accum_steps 8 \
    --max_seq_len 6144 \
    --wandb

# arlong repeat demonstration
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_arlong_repeatdemon \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --grad_accum_steps 8 \
    --max_seq_len 6144 \
    --repeat_demonstration \
    --wandb

# Submitted batch job 57179117
# Submitted batch job 57179118