# tested training and evaluating model by making sure gen outputs are exactly same, work for 16 and 32 bits
#        works for ntokens attentioncuttoff prevprogram all that stuff



# overfit noprogram
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_copy/train.py \
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
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_copy/evaluate.py \
    --tag test \
    --weight_dir test \
    --weight_epoch 1 \
    --data_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --ntokens 32 \
    --attention_cutoff \
    --attend_prev_programs




# overfit noprogram (shortcontext or not)
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_copy/train.py \
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
    --attention_reduction_ratio 0.5 \
    --short_context

# evaluate noprogram (shortcontext or not)
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_copy/evaluate.py \
    --tag test \
    --weight_dir test \
    --weight_epoch 1 \
    --data_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --attention_reduction_ratio 0.5 \
    --short_context
