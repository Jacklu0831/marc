# try to save and load autoregressive model

# overfit1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --tag test \
    --train_data_dir ./data/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit/training \
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
    --no_flash_attn

# evaluate the above
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/evaluate.py \
    --tag test \
    --weight_dir test \
    --weight_epoch 1 \
    --data_dir ./data/re-arc/arc_original_debug_overfit/training

# now ttt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/ttt.py \
    --tag test \
    --weight_dir test \
    --weight_epoch 1 \
    --data_dir ./data/re-arc/arc_original_debug_overfit2_ttt/training \
    --max_samples_per_task 4 \
    --grad_accum_steps 1 \
    --optimizer sgd \
    --debug_no_aug \
    --max_grad_norm 1e2 \
    --lr 1e-3 \
    --num_epochs 200 \
    --log_every 1 \
    --save_epochs 200

# ok now evaluate ttt (only gets 1 out of 2)
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/evaluate.py \
    --tag test \
    --weight_dir test \
    --weight_epoch 1 \
    --ttt_weight_dir ttt_test_test \
    --ttt_weight_epoch 200 \
    --data_dir ./data/re-arc/arc_original_debug_overfit2_ttt/training
