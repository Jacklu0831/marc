# python make_sbatch.py --ngpu 4 --time 48 --bash_files bash_commands/0218_autoregressive/0218_16_increaselr.sh --burst --ncpu 4

# ar gradaccum8 lr4e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_gradaccum8_lr4e-4 \
    --train_batch_size 1 \
    --grad_accum_steps 8 \
    --lr_embedding 4e-5 \
    --lr_program 4e-4 \
    --lr_prior 4e-4 \
    --lr_other 4e-4 \
    --eval_batch_size 32 \
    --num_workers 4 \
    --wandb

# ar gradaccum8 lr4e-4 warmup2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_gradaccum8_lr4e-4_warmup2 \
    --train_batch_size 1 \
    --grad_accum_steps 8 \
    --lr_embedding 4e-5 \
    --lr_program 4e-4 \
    --lr_prior 4e-4 \
    --lr_other 4e-4 \
    --eval_batch_size 32 \
    --num_workers 4 \
    --warmup_epoch 2 \
    --wandb

# ar gradaccum16 lr8e-4 warmup2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_gradaccum16_lr8e-4_warmup2 \
    --train_batch_size 1 \
    --grad_accum_steps 16 \
    --lr_embedding 8e-5 \
    --lr_program 8e-4 \
    --lr_prior 8e-4 \
    --lr_other 8e-4 \
    --eval_batch_size 32 \
    --num_workers 4 \
    --warmup_epoch 2 \
    --wandb

# Submitted batch job 33390
# Submitted batch job 33391
# Submitted batch job 33392

# gradaccum16 8e-4 warmup1 lossjumped
