# all tested on rtx8000 because greene sucks
# tested training and evaluating model by making sure gen outputs are exactly same, work for 16 and 32 bits
#        works for ntokens attentioncuttoff prevprogram all that stuff
# tested that eval and ttt load the exact same model, but did not ensure program embedding and such are loaded properly
# tested ttt model is refreshed after each eval task training


# overfit noprogram
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --debug_no_resume \
    --no_flash_attn \
    --tag test \
    --log_every 1 \
    --train_data_dir ./data/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --eval_epochs 1 \
    --num_epochs 1 \
    --samples_per_epoch 100 \
    --lr_embedding 1e-2 \
    --lr_program 1e-2 \
    --lr_prior 1e-2 \
    --lr_other 1e-2 \
    --ntokens 32 \
    --attention_cutoff \
    --attend_prev_programs

# evaluate noprogram
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --tag test \
    --weight_dir test \
    --weight_epoch 1 \
    --data_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --ntokens 32 \
    --attention_cutoff \
    --attend_prev_programs

# now ttt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/ttt.py \
    --tag test \
    --weight_dir test \
    --weight_epoch 1 \
    --data_dir ./data/re-arc/arc_original_debug_overfit2_ttt/training \
    --max_samples_per_task 4 \
    --num_epochs 1 \
    --save_epochs 1 \
    --log_every 1 \
    --no_flash_attn



# overfit arlongcache (shortcontext or not)
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --debug_no_resume \
    --no_flash_attn \
    --tag test \
    --log_every 1 \
    --train_data_dir ./data/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --eval_epochs 1 \
    --num_epochs 1 \
    --samples_per_epoch 32 \
    --lr_embedding 1e-2 \
    --lr_program 1e-2 \
    --lr_prior 1e-2 \
    --lr_other 1e-2 \
    --attention_reduction_ratio 0.5 \
    --short_context

# evaluate arlongcache (shortcontext or not)
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/evaluate.py \
    --tag test \
    --weight_dir test \
    --weight_epoch 1 \
    --data_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --attention_reduction_ratio 0.5 \
    --short_context

# now arlongcache
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/ttt.py \
    --tag test \
    --weight_dir test \
    --weight_epoch 1 \
    --data_dir ./data/re-arc/arc_original_debug_overfit2_ttt/training \
    --max_samples_per_task 4 \
    --num_epochs 1 \
    --save_epochs 1 \
    --log_every 1 \
    --no_flash_attn






# second part
# tested that arlongcache ttt has same loss as noprogram (when arlongcache ntokens=0)

# first obtain a noprogram checkpoint WITHOUT training
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_noprogram_copy/train.py \
    --train_data_dir ./data/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --no_bos \
    --debug_no_resume \
    --no_flash_attn \
    --tag test_noprogram \
    --num_epochs 1 \
    --eval_epochs 1 \
    --samples_per_epoch 8 \
    --lr_embedding 0.0 \
    --lr_program 0.0 \
    --lr_prior 0.0 \
    --lr_other 0.0 \
    --max_seq_len 2048

# then ttt noprogram to check loss (try d8 and extra)
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_noprogram_copy/ttt.py \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --no_bos \
    --select_tasks_path task_info_selected.csv \
    --weight_dir test_noprogram \
    --weight_epoch 1 \
    --tag test \
    --num_epochs 1 \
    --save_epochs 1 \
    --aug_type extra \
    --no_flash_attn




# first obtain a arlongcache checkpoint WITHOUT training
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_autoregressive_longcontext_caching_copy/train.py \
    --train_data_dir ./data/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --no_bos \
    --token_weighted_loss \
    --debug_no_resume \
    --no_flash_attn \
    --tag test_arlongcache \
    --num_epochs 1 \
    --eval_epochs 1 \
    --samples_per_epoch 8 \
    --lr_embedding 0.0 \
    --lr_program 0.0 \
    --lr_prior 0.0 \
    --lr_other 0.0 \
    --max_seq_len 2048 \
    --ntokens 0

# then ttt arlongcache to check loss (try d8 and extra)
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_autoregressive_longcontext_caching_copy/ttt.py \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --no_bos \
    --token_weighted_loss \
    --select_tasks_path task_info_selected.csv \
    --weight_dir test_arlongcache \
    --weight_epoch 1 \
    --tag test \
    --num_epochs 1 \
    --save_epochs 1 \
    --aug_type extra \
    --ntokens 0 \
    --no_flash_attn
