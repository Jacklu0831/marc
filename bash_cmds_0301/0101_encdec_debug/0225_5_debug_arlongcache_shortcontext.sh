# test that shortcontext overfits overfit1 and overfit4
# and is same as original AR (all pass)

# try to overfit1 and overfit4 (overfit4 is poor)
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0310/train.py \
    --tag test \
    --train_data_dir ./data/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 500 \
    --lr_embedding 1e-3 \
    --lr_program 1e-3 \
    --lr_prior 1e-3 \
    --lr_other 1e-3 \
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
    --debug_no_resume \
    --train_pad_side left \
    --short_context \
    --samples_per_epoch 8

# ar long cache that collapses to AR
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_autoregressive_longcontext_caching_0316/train.py \
    --tag test \
    --train_data_dir ./data/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --no_flash_attn \
    --token_weighted \
    --debug_no_resume \
    --ntokens 4 \
    --lr_embedding 0.0 \
    --lr_program 0.0 \
    --lr_prior 0.0 \
    --lr_other 0.0 \
    --eval_batch_size 100 \
    --short_context \
    --grad_accum_steps 1 \
    --train_batch_size 1 \
    --samples_per_epoch 1 \
    --eval_epochs 1

# actual AR
accelerate launch --main_process_port $MASTER_PORT depre/encoder_decoder_autoregressive_0305/train.py \
    --tag test \
    --train_data_dir ./data/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --no_flash_attn \
    --token_weighted \
    --debug_no_resume \
    --ntokens 4 \
    --lr_embedding 0.0 \
    --lr_program 0.0 \
    --lr_prior 0.0 \
    --lr_other 0.0 \
    --eval_batch_size 100 \
    --grad_accum_steps 1 \
    --train_batch_size 1 \
    --samples_per_epoch 1 \
    --eval_epochs 1
