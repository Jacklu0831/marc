# tested reduction overfit on 1 and 4
# tested left and right pad same result for reduction
# tested ratio 1 is same as original implementation, just sanity check for back compat
# tested generation actually properly works by just comparing 0token to baseline generation
# tested all of these for flashattn or not
# tested again that ntoken0 becomes baseline, shortcontext becomes AR

# NOTE: llama1b actually has 32 query heads and only 8 KV heads
# NOTE: modifying keys in-place does not affect the return value of training, but affects generation
# NOTE: wanted to test reduction ratio 0.0 is equal to shortcontext, but its not because it has attention to past programs
# NOTE: fitting on diffnumpair is fine on one gpu, but cant for mutligpu because reduce requires SAME compute graph due to debug_fixed_order

# TODO now WITH THE BIG FIX:
# AR and noprogram baseline, numpair3, noaug, extraaug, extraaugsingle, 100/200 tasks
# AR ntoken search 2, 4, 8, 16, 32
# AR everything: invarloss, programloss, consistencyloss, fullttention dropout, full/demonstration dropout, shortcontext (ntoken search)


# make generate work with kv cache, finally works
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --train_data_dir ./data/re-arc/train_data_debug_overfit2_diffnumpair/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit2_diffnumpair/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit2_diffnumpair/training \
    --debug_no_resume \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag test \
    --attention_reduction_ratio 1.0 \
    --no_flash_attn \
    --samples_per_epoch 2 \
    --grad_accum_steps 1

# make sure new way to generate with kv cache is same as original, change train.py dir
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_autoregressive_longcontext_caching_0316/train.py \
    --debug_no_resume \
    --tag test \
    --no_tf32 \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --debug \
    --lr_embedding 0.0 \
    --lr_program 0.0 \
    --lr_prior 0.0 \
    --lr_other 0.0 \
    --attention_reduction_ratio 0.5 \
    --no_flash_attn

    --lr_embedding 1e-3 \
    --lr_program 1e-3 \
    --lr_prior 1e-3 \
    --lr_other 1e-3

# realized that models sample data the same way for each epoch, existential level crisis moment
# fixed both AR and baseline, checked they are equal when ntoken=0 again
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --debug_no_resume \
    --tag test \
    --lr_embedding 0.0 \
    --lr_program 0.0 \
    --lr_prior 0.0 \
    --lr_other 0.0 \
    --samples_per_epoch 16 \
    --train_batch_size 2 \
    --num_workers 2 \
    --grad_accum_steps 1 \
    --no_flash_attn \
    --eval_epochs 4 \
    --log_every 10000000

# test flashattn or not (cant be float32 due to flashattn so) -> ~0.1-1% difference in celoss, should be fine?
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0316/train.py \
    --debug_no_resume \
    --tag test \
    --lr_embedding 0.0 \
    --lr_program 0.0 \
    --lr_prior 0.0 \
    --lr_other 0.0 \
    --samples_per_epoch 64 \
    --train_batch_size 1 \
    --grad_accum_steps 1 \
    --debug \
    --no_tf32 \
    --attention_reduction_ratio 0.5 \
    --no_flash_attn

# test if eager attention is same as sdpa (eager uses more memory, so half batchsize), float16 for memory
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0316/train.py \
    --debug_no_resume \
    --tag test \
    --debug \
    --lr_embedding 0.0 \
    --lr_program 0.0 \
    --lr_prior 0.0 \
    --lr_other 0.0 \
    --max_seq_len 1024 \
    --no_flash_attn \
    --attention_reduction_ratio 0.5 \
    --demonstration_attention_dropout 0.1

# try to fit ar long cache on overfit WITH REDUCTION OR DROPOUT (it works yay)
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --debug_no_resume \
    --tag test \
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
    --attention_reduction_ratio 0.3 \
    --partial_demonstration_dropout 0.3 \
    --full_demonstration_dropout 0.3 \
    --train_data_dir ./data/re-arc/train_data_debug_overfit4/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit4/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit4/training

    --train_data_dir ./data/re-arc/train_data_debug_overfit2_diffnumpair/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit2_diffnumpair/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit2_diffnumpair/training \
    --train_batch_size 1 \
    --samples_per_epoch 250 \

    --train_data_dir ./data/re-arc/train_data_debug_overfit/tasks \
    --eval_train_dir ./data/re-arc/arc_original_debug_overfit/training \
    --eval_eval_dir ./data/re-arc/arc_original_debug_overfit/training \