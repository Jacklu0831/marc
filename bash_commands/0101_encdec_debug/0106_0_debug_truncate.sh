# original overfit
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0218/train.py \
    --tag test \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 500 \
    --lr_embedding 1e-2 \
    --lr_program 1e-2 \
    --lr_prior 1e-2 \
    --lr_other 1e-2 \
    --debug_fixed_order \
    --max_grad_norm 1e2 \
    --num_workers 0 \
    --optimizer sgd \
    --min_num_pair 5 \
    --max_num_pair 5 \
    --no_color_permute \
    --no_pair_permute \
    --no_d8 \
    --no_train_original \
    --log_every 1 \
    --grad_accum_steps 1

# truncate overfit
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_truncated/train.py \
    --tag test \
    --train_data_dir /scratch/yl11330/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir /scratch/yl11330/re-arc/arc_original_debug_overfit/training \
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 500 \
    --lr_embedding 1e-2 \
    --lr_program 1e-2 \
    --lr_prior 1e-2 \
    --lr_other 1e-2 \
    --debug_fixed_order \
    --max_grad_norm 1e2 \
    --num_workers 0 \
    --optimizer sgd \
    --min_num_pair 5 \
    --max_num_pair 5 \
    --no_color_permute \
    --no_pair_permute \
    --no_d8 \
    --no_train_original \
    --log_every 1 \
    --grad_accum_steps 1 \
    --truncation_steps 100

# original
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217_temp/train.py \
    --lr_scheduler constant \
    --warmup_epoch 1 \
    --tag test \
    --train_batch_size 2 \
    --grad_accum_steps 8 \
    --trainable_nbit 32

# truncate
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_truncated/train.py \
    --lr_scheduler constant \
    --warmup_epoch 1 \
    --tag test \
    --train_batch_size 2 \
    --grad_accum_steps 8 \
    --truncation_steps 100 \
    --trainable_nbit 32

# memory test
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_truncated/train.py \
    --lr_scheduler constant \
    --warmup_epoch 1 \
    --tag test \
    --train_batch_size 8 \
    --grad_accum_steps 2 \
    --truncation_steps 2 \
    --samples_per_epoch 64 \
    --debug_len 1867



# celoss:
# 5.797219276428223
# 5.797219276428223
# 7.425548553466797
# 7.3504486083984375

# 5.797219276428223
# 5.797219276428223
# 7.425548553466797
# 7.3504486083984375

# gradnorm:
# 53059.67578125
# 53059.67578125
# 3081.8837890625
# 1769.4212646484375

# 53059.67578125
# 53059.67578125
# 3081.8837890625
# 1769.4212646484375