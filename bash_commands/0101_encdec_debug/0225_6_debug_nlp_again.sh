# time to truly debug noprogram and arlongcache for nlp

# just trying to run gpt2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --debug_no_resume \
    --lr_scheduler constant \
    --tag test \
    --model_name gpt2 \
    --no_lora \
    --grad_accum_steps 1 \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --debug

# just trying to run llama1b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --debug_no_resume \
    --lr_scheduler constant \
    --tag test \
    --model_name llama1b \
    --grad_accum_steps 1 \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --debug

# overfit gpt2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --debug_no_resume \
    --lr_scheduler constant \
    --tag test \
    --model_name gpt2 \
    --no_lora \
    --debug_no_resume \
    --debug_fixed_order \
    --config_file MetaICL/config/toy.json \
    --data_dir MetaICL/data_toy \
    --samples_per_epoch 100 \
    --lr_embedding 0.0 \
    --lr_other 0.0

# overfit llama1b
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --debug_no_resume \
    --lr_scheduler constant \
    --tag test \
    --model_name llama1b \
    --debug_no_resume \
    --debug_fixed_order \
    --config_file MetaICL/config/toy.json \
    --data_dir MetaICL/data_toy \
    --samples_per_epoch 100 \
    --lr_embedding 0.0 \
    --lr_other 0.0
