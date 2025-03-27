# test gs for arlongcache short/longcontext as well as noprogram

# very high lr of 10.0 and many iters of 1000 needed for a model from scratch, but think its learning correctly
# for longcontext: easily fit with gstrainpastkv, very hard to fit without
# for shortcontext: same but less hard to fit without gstrainpastkv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0324/train.py \
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
    --eval_batch_size 8 \
    --train_gs_lr 10.0 \
    --train_gs_batch_size 100000 \
    --train_gs_optimizer sgd \
    --train_gs_max_grad_norm 1e8 \
    --train_gs_iters 250 \
    --train_gs_take_best \
    --train_gs_train_past_kv \
    --short_context

# try float32 to ensure padleft and padright are the same
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_autoregressive_longcontext_caching_0324/train.py \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --no_flash_attn \
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
    --eval_batch_size 8 \
    --train_gs_lr 10.0 \
    --train_gs_batch_size 100000 \
    --train_gs_optimizer sgd \
    --train_gs_max_grad_norm 1e8 \
    --train_gs_iters 250 \
    --train_gs_take_best \
    --train_gs_train_past_kv \
    --debug_pad_len 5 \
    --train_pad_side right \
    --kv_pad_side right \
    --short_context

# let's try it with noprogram past key value cache
# tested gs works on overfit2_ttt
# tested gs is same for padleft and right
# tested gs is same as arlongcache with token0
# also tried checked that if two-step inference is done but gt_iters = 0 (comment out gt code in evaluate), same out tokens for long/shortcontext
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_0323/train.py \
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
    --eval_batch_size 8 \
    --train_gs_lr 10.0 \
    --train_gs_batch_size 100000 \
    --train_gs_optimizer sgd \
    --train_gs_max_grad_norm 1e8 \
    --train_gs_iters 500 \
    --train_gs_take_best

# try float32 to ensure padleft and padright are the same (not necessarily for generation because it requires left, but for gs loss)
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_noprogram_0323/train.py \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --no_flash_attn \
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
    --eval_batch_size 8 \
    --train_gs_lr 10.0 \
    --train_gs_batch_size 100000 \
    --train_gs_optimizer sgd \
    --train_gs_max_grad_norm 1e8 \
    --train_gs_iters 250 \
    --train_gs_take_best \
    --debug_random_pad \
    --pad_side left

# arlongcache ntoken0 to compare with baseline implementation
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_autoregressive_longcontext_caching_0324/train.py \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --no_flash_attn \
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
    --eval_batch_size 8 \
    --train_gs_lr 10.0 \
    --train_gs_batch_size 100000 \
    --train_gs_optimizer sgd \
    --train_gs_max_grad_norm 1e8 \
    --train_gs_iters 250 \
    --train_gs_take_best \
    --train_gs_train_past_kv \
    --train_pad_side right \
    --kv_pad_side right \
    --debug_pad_len 100 \
    --ntokens 0