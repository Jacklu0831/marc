# test if shortcontext truncated works
# test if code is backward compatible (same outputs as before if no cache)
# should try nod8, nocoloraug, lossonfirst or not

# check if cache is consistency across multigpu on rtx8000
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0324/train.py \
    --debug_no_resume \
    --tag test \
    --train_data_dir ./data/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --samples_per_epoch 32 \
    --min_num_pair 2 \
    --max_num_pair 2 \
    --log_every 1 \
    --short_context \
    --loss_on_first \
    --cache_size_per_task 100 \
    --prior_embed_ratio 0.5 \
    --no_flash_attn

# making sure backward compatibility (switch between two train scripts), thankfully it is
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_autoregressive_longcontext_caching_0324/train.py \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_flash_attn \
    --no_tf32 \
    --debug_no_resume \
    --tag test \
    --samples_per_epoch 8

# try overfitting on rtx8000, overfit1 works!
# overfit4 gives it a hard time, only gets 1/7 correct, worrying
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0324/train.py \
    --debug_no_resume \
    --tag test \
    --train_data_dir ./data/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit/training \
    --no_pair_permute \
    --no_d8 \
    --no_color_permute \
    --no_train_original \
    --eval_epochs 1 \
    --num_epochs 100 \
    --samples_per_epoch 500 \
    --lr_embedding 1e-3 \
    --lr_program 1e-3 \
    --lr_prior 1e-3 \
    --lr_other 1e-3 \
    --min_num_pair 2 \
    --max_num_pair 2 \
    --max_grad_norm 1e2 \
    --num_workers 0 \
    --optimizer sgd \
    --grad_accum_steps 1 \
    --train_pad_side left \
    --short_context \
    --loss_on_first \
    --cache_size_per_task 100 \
    --prior_embed_ratio 0.1 \
    --no_flash_attn

# profile memory, keep cache size minimal, allocate memory of max cache size see if OOM
# tested on rtx8000
# 4byte x 2048hiddendim x 4token x 400task x 100cachepertask x 8basicaug = 10.5GB
# saw available mem go from 343 to 333GB
# when set 400 cache per task, saw mem go from 343 to DEATH, 32GB gpu is not enough
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0324/train.py \
    --debug_no_resume \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag test \
    --train_data_dir ./data/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit/training \
    --no_pair_permute \
    --no_d8 \
    --no_color_permute \
    --no_train_original \
    --min_num_pair 2 \
    --max_num_pair 2 \
    --short_context \
    --loss_on_first \
    --cache_size_per_task 400 \
    --prior_embed_ratio 0.1 \
    --samples_per_epoch 32 \
    --no_color_permute \
    --no_flash_attn \
    --log_every 1

# finally, since minnumpair is now 2, lets try to increase train batch size
# eval is same as without cache, so no change to its params
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0324/train.py \
    --debug_no_resume \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag test \
    --short_context \
    --loss_on_first \
    --cache_size_per_task 50 \
    --prior_embed_ratio 0.1 \
    --no_color_permute \
    --min_num_pair 2 \
    --max_num_pair 2 \
    --samples_per_epoch 40000 \
    --train_batch_size 4