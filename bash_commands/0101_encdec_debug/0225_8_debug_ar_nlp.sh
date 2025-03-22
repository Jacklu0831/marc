# time to truly debug noprogram and arlongcache for nlp
# test1: ensure arlongcache padleft and right same results (with all the dropout and reduction and stuff, shortcontext or not)
# test2: ensure arlongcache ntoken=0 == baseline (pad left and pad right for both, all same model loss)

for x1, x2 in zip(saved, saved_all_programs): print((x1-x2).abs().mean())


# AR no toy, padleftvsright, losstype
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_autoregressive_longcontext_caching_nlp/train.py \
    --debug \
    --max_seq_len 2048 \
    --log_every 1 \
    --tag test \
    --eval_min_num_pair 2 \
    --no_flash_attn \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --token_weighted_loss \
    --kv_pad_side left \
    --pad_side left \
    --loss_type exclude_first \
    --lr_embedding 0.1 \
    --lr_program 0.1 \
    --lr_prior 0.1 \
    --lr_other 0.1 \
    --ntokens 0 \
    --eval_pretrained

# noprogram
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_noprogram_nlp/train.py \
    --debug \
    --max_seq_len 2048 \
    --log_every 1 \
    --tag test \
    --eval_min_num_pair 2 \
    --no_flash_attn \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --loss_type exclude_first \
    --lr_embedding 0.1 \
    --lr_program 0.1 \
    --lr_prior 0.1 \
    --lr_other 0.1 \
    --eval_pretrained

# ok now that AR normal training works, debug shortcontext, wait it works
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_autoregressive_longcontext_caching_nlp/train.py \
    --debug \
    --max_seq_len 2048 \
    --log_every 1 \
    --tag test \
    --eval_min_num_pair 2 \
    --no_flash_attn \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --token_weighted_loss \
    --kv_pad_side right \
    --pad_side right \
    --loss_type exclude_first \
    --lr_embedding 0.0 \
    --lr_program 0.0 \
    --lr_prior 0.0 \
    --lr_other 0.0 \
    --ntokens 0 \
    --short_context

# now debug shortcontext for toy (wait it works too lmao)
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_autoregressive_longcontext_caching_nlp/train.py \
    --debug_no_resume \
    --debug_fixed_order \
    --config_file MetaICL/config/toy.json \
    --data_dir MetaICL/data_toy \
    --max_seq_len 2048 \
    --log_every 1 \
    --tag test \
    --eval_min_num_pair 2 \
    --no_flash_attn \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --token_weighted_loss \
    --kv_pad_side left \
    --pad_side left \
    --loss_type exclude_first \
    --lr_embedding 0.0 \
    --lr_program 0.0 \
    --lr_prior 0.0 \
    --lr_other 0.0 \
    --ntokens 0 \
    --short_context