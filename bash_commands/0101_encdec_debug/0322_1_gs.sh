# tested gs for shortcontext or not

# overfit arlongcache (shortcontext or not)
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/train.py \
    --no_bos \
    --debug_no_resume \
    --tag test \
    --log_every 1 \
    --train_data_dir ./data/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit2_ttt/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit2_ttt/training \
    --no_train_original \
    --debug_fixed_order \
    --eval_epochs 1 \
    --num_epochs 1 \
    --samples_per_epoch 8 \
    --eval_batch_size 1 \
    --eval_gs_lr 1e-2 \
    --eval_gs_batch_size 100000 \
    --eval_gs_optimizer sgd \
    --eval_gs_max_grad_norm 1e8 \
    --eval_gs_iters 100 \
    --eval_gs_take_best \
    --short_context

# use evaluate.py with a decent checkpoint instead
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0323/evaluate.py \
    --tag test \
    --weight_dir 0317_arlongcache_100task \
    --weight_epoch 22 \
    --leave_ns 0 \
    --data_dir ./data/re-arc/arc_original_debug_overfit2_ttt/training \
    --batch_size 1 \
    --gs_lr 1e-2 \
    --gs_batch_size 100000 \
    --gs_optimizer sgd \
    --gs_max_grad_norm 1e8 \
    --gs_iters 100 \
    --gs_take_best