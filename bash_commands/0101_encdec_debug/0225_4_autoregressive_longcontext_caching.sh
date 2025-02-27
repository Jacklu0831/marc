# try to save and load autoregressive model

# overfit1 with fsq (had to use rtx8000)
# use trainbatchsize 1 for diffnumpair
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_autoregressive_longcontext_caching/train.py \
    --tag test \
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
    --no_color_permute \
    --no_pair_permute \
    --no_d8 \
    --no_train_original \
    --log_every 1 \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --token_weighted_loss \
    --train_batch_size 1 \
    --grad_accum_steps 1 \
    --no_flash_attn \
    --debug_random_pad \
    --train_data_dir ./data/re-arc/train_data_debug_overfit2_diffnumpair/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit2_diffnumpair/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit2_diffnumpair/training

    --train_data_dir ./data/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit/training
    --train_data_dir ./data/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit4/training


accelerate launch --mixed_precision bf16 --main_process_port $MASTER_PORT encoder_decoder_autoregressive_longcontext_caching/train.py     --tag test     --eval_epochs 1     --num_epochs 100     --samples_per_epoch 500     --lr_embedding 1e-3     --lr_program 1e-3     --lr_prior 1e-3     --lr_other 1e-3     --debug_fixed_order     --max_grad_norm 1e2     --num_workers 0     --optimizer sgd     --min_num_pair 5     --max_num_pair 5     --no_color_permute     --no_pair_permute     --no_d8     --no_train_original     --log_every 1          --token_weighted_loss     --train_batch_size 2     --grad_accum_steps 1       --train_data_dir ./data/re-arc/train_data_debug_overfit/tasks     --eval_train_dir ./data/re-arc/arc_original_debug_overfit/training     --eval_eval_dir ./data/re-arc/arc_original_debug_overfit/training --debug_random_pad --no_flash_attn
# overfit1 trainpadside=right works very well
# overfit1 trainpadside=right with randomdebug <- need to implement custom attention and figure out if it works with flashattn too
