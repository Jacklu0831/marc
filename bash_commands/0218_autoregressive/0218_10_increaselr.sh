# python make_sbatch.py --ngpu 4 --time 48 --bash_files bash_commands/0218_autoregressive/0218_10_increaselr.sh --burst

# ar gradaccum8 lr3e-4 warmup2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_gradaccum8_lr3e-4_warmup2 \
    --train_batch_size 1 \
    --grad_accum_steps 8 \
    --lr_embedding 3e-5 \
    --lr_program 3e-4 \
    --lr_prior 3e-4 \
    --lr_other 3e-4 \
    --warmup_epoch 2 \
    --eval_batch_size 32 \
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
    --warmup_epoch 2 \
    --eval_batch_size 32 \
    --wandb

# ar gradaccum8 lr3e-4 warmup4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_gradaccum8_lr3e-4_warmup4 \
    --train_batch_size 1 \
    --grad_accum_steps 8 \
    --lr_embedding 3e-5 \
    --lr_program 3e-4 \
    --lr_prior 3e-4 \
    --lr_other 3e-4 \
    --warmup_epoch 4 \
    --eval_batch_size 32 \
    --wandb

# ar gradaccum8 lr4e-4 warmup4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_gradaccum8_lr4e-4_warmup4 \
    --train_batch_size 1 \
    --grad_accum_steps 8 \
    --lr_embedding 4e-5 \
    --lr_program 4e-4 \
    --lr_prior 4e-4 \
    --lr_other 4e-4 \
    --warmup_epoch 4 \
    --eval_batch_size 32 \
    --wandb

# Submitted batch job 33346
# Submitted batch job 33347
# Submitted batch job 33348
# Submitted batch job 33349