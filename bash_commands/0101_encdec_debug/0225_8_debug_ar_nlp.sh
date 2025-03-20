# time to truly debug noprogram and arlongcache for nlp

# AR
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --debug_no_resume \
    --lr_scheduler constant \
    --tag test \
    --model_name llama1b \
    --debug_fixed_order \
    --config_file MetaICL/config/toy.json \
    --data_dir MetaICL/data_toy \
    --samples_per_epoch 100 \
    --eval_pretrained

# noprogram
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --debug_no_resume \
    --lr_scheduler constant \
    --tag test \
    --model_name llama1b \
    --debug_fixed_order \
    --config_file MetaICL/config/toy.json \
    --data_dir MetaICL/data_toy \
    --samples_per_epoch 100 \
    --eval_pretrained
