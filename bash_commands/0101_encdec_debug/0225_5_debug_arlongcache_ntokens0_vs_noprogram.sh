# float16 sucks, even with notf32 and noflashattn, it has very significant error
# ended up realizing noprogram cannot generate when pad left, so lets make sure gen pad side == left always
# ended up in a rabbit hole of debugging train pad left vs right, they are finally the same now for model_loss and evaluate/generate even for diffnumpairs2
# ended up checking noprogram pad left vs pad right too, they get same for model_loss (in reality, we keep padside left for generate)
# ended up ensuring noprogram ntoken0 and ntoken-1 are the exact same in training and evaluation
# finally proved AR ntoken0 == noprogram ntoken0/-1, need nobos
# TODO: thoroughly test noprogram with tokens, maybe pad left vs right is not the same

# TODO runs: rerun AR and noprogram with and without nobos, full precision AR nobos and noprogram nobos, ar shortcontext nobos

# AR that collapses back to baseline
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_autoregressive_longcontext_caching/train.py \
    --train_data_dir ./data/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --tag test \
    --log_every 1 \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --token_weighted_loss \
    --no_flash_attn \
    --ntokens 0 \
    --debug_no_resume \
    --debug \
    --no_bos \
    --train_pad_side left \
    --no_color_permute \
    --no_pair_permute \
    --lr_embedding 0.0 \
    --lr_program 0.0 \
    --lr_prior 0.0 \
    --lr_other 0.0 \
    --eval_batch_size 100

# actual baseline
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_noprogram/train.py \
    --train_data_dir ./data/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --tag test \
    --log_every 1 \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --no_flash_attn \
    --debug_no_resume \
    --debug \
    --no_bos \
    --no_color_permute \
    --no_pair_permute \
    --lr_embedding 0.0 \
    --lr_program 0.0 \
    --lr_prior 0.0 \
    --lr_other 0.0 \
    --eval_batch_size 100



# baseline with ntoken0 left (only check model loss)
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_noprogram/train.py \
    --train_data_dir ./data/re-arc/train_data_debug_overfit2_diffnumpair/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit2_diffnumpair/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit2_diffnumpair/training \
    --tag test \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --no_flash_attn \
    --ntokens 16 \
    --debug_no_resume \
    --debug \
    --attention_cutoff \
    --attend_prev_programs \
    --pad_side left

# baseline with ntoken0 right (only check model loss)
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_noprogram/train.py \
    --train_data_dir ./data/re-arc/train_data_debug_overfit2_diffnumpair/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit2_diffnumpair/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit2_diffnumpair/training \
    --tag test \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --no_flash_attn \
    --ntokens 16 \
    --debug_no_resume \
    --debug \
    --attention_cutoff \
    --attend_prev_programs \
    --pad_side right