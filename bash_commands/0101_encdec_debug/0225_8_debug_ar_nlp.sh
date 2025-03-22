# time to truly debug noprogram and arlongcache for nlp
# test1: ensure arlongcache padleft and right same results (with all the dropout and reduction and stuff, shortcontext or not)
# test2: ensure arlongcache ntoken=0 == baseline

# AR
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_autoregressive_longcontext_caching_nlp/train.py \
    --debug_no_resume \
    --log_every 1 \
    --lr_scheduler constant \
    --tag test \
    --debug_fixed_order \
    --config_file MetaICL/config/toy.json \
    --data_dir MetaICL/data_toy \
    --samples_per_epoch 100 \
    --eval_min_num_pair 2 \
    --no_flash_attn \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --token_weighted_loss \
    --kv_pad_side left \
    --pad_side left \
    --short_context \
    --debug \
    --train_batch_size 2 \
    --loss_type only_last \
    --lr_embedding 0.0 \
    --lr_program 0.0 \
    --lr_prior 0.0 \
    --lr_other 0.0

    --eval_pretrained \
    --attention_reduction_ratio 0.5 \
    --partial_demonstration_dropout 0.5 \
    --full_demonstration_dropout 0.5 \

# noprogram
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_noprogram_nlp/train.py \
    --debug_no_resume \
    --log_every 1 \
    --lr_scheduler constant \
    --tag test \
    --debug_fixed_order \
    --config_file MetaICL/config/toy.json \
    --data_dir MetaICL/data_toy \
    --samples_per_epoch 100 \
    --num_workers 0 \
    --eval_min_num_pair 2 \
    --no_flash_attn \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32