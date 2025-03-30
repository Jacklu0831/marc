# when working on toy tasks for noprogram, realized some deep problems with ntokens that I think were avoided by previous runs
# but its good to keep up the code quality anyway:

# - if flashattn is disabled, sdpa is used and llamamodel ignores a causal attention mask -> cut off attention doesnt work
# - if training sample has different number of pairs in batch, padding is incorrect
# - generally seriously debug noprogram ntokens, might be bug that cause bad performance

# tested padleft == padright for all possible configurations
# tested padleft generation with with == without random debug pad
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_noprogram_0327/train.py \
    --samples_per_epoch 6 \
    --grad_accum_steps 3 \
    --train_batch_size 2 \
    --eval_epochs 1 \
    --tag test \
    --log_every 1 \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --no_flash_attn \
    --debug_no_resume \
    --no_bos \
    --lr_embedding 0.0 \
    --lr_program 0.0 \
    --lr_prior 0.0 \
    --lr_other 0.0 \
    --eval_batch_size 100 \
    --pad_side left \
    --ntokens 128 \
    --attention_cutoff \
    --attend_prev_programs \
    --debug_random_pad \
    --eval_batch_size 8


    --train_data_dir ./data/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit4/training \