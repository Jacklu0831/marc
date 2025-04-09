# implemented noprogram evaluation script, lets test it
# only tested without ntokens, noprogram nlp with tokens is not implemented
# float16 output losses have slight deviation, float32 doesnt
# tried to get dummy loss from end train.py and just after loading in eval, 0.0 difference

# overfit noprogram
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --debug_no_resume \
    --lr_scheduler constant \
    --tag test \
    --config_file MetaICL/config/toy.json \
    --data_dir MetaICL/data_toy \
    --lr_embedding 0.001 \
    --lr_other 0.001 \
    --num_epochs 1 \
    --eval_epochs 1 \
    --samples_per_epoch 250

# evaluate noprogram
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/evaluate.py \
    --tag test \
    --weight_dir test \
    --weight_epoch 1 \
    --config_file MetaICL/config/toy.json \
    --data_dir MetaICL/data_toy \
    --flash_attn



# overfit noprogram precise
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_noprogram_nlp/train.py \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_flash_attn \
    --no_tf32 \
    --debug_no_resume \
    --lr_scheduler constant \
    --tag test \
    --config_file MetaICL/config/toy.json \
    --data_dir MetaICL/data_toy \
    --lr_embedding 0.001 \
    --lr_other 0.001 \
    --num_epochs 1 \
    --eval_epochs 1 \
    --samples_per_epoch 100

# evaluate noprogram precise
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_noprogram_nlp/evaluate.py \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --tag test \
    --weight_dir test \
    --weight_epoch 1 \
    --config_file MetaICL/config/toy.json \
    --data_dir MetaICL/data_toy










# implemented arlongcache evaluation script, lets test it
# interestingly, EXACTLY the same behavior seen in noprogram nlp

# overfit arlongcache
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_nlp/train.py \
    --debug_no_resume \
    --lr_scheduler constant \
    --tag test \
    --config_file MetaICL/config/toy.json \
    --data_dir MetaICL/data_toy \
    --lr_embedding 0.001 \
    --lr_other 0.001 \
    --num_epochs 1 \
    --eval_epochs 1 \
    --samples_per_epoch 64

# evaluate arlongcache
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_nlp/evaluate.py \
    --tag test \
    --weight_dir test \
    --weight_epoch 1 \
    --config_file MetaICL/config/toy.json \
    --data_dir MetaICL/data_toy \
    --flash_attn




# overfit arlongcache precise
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_autoregressive_longcontext_caching_nlp/train.py \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_flash_attn \
    --no_tf32 \
    --debug_no_resume \
    --lr_scheduler constant \
    --tag test2 \
    --config_file MetaICL/config/toy.json \
    --data_dir MetaICL/data_toy \
    --lr_embedding 0.001 \
    --lr_other 0.001 \
    --num_epochs 1 \
    --eval_epochs 1 \
    --samples_per_epoch 64

# evaluate arlongcache precise
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_autoregressive_longcontext_caching_nlp/evaluate.py \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --tag test \
    --weight_dir test2 \
    --weight_epoch 1 \
    --config_file MetaICL/config/toy.json \
    --data_dir MetaICL/data_toy
