# test gs for arlongcache short/longcontext as well as noprogram NOW FOR NLP

# do a quick overfit run just to check, ok it works (goes from 0.5 on both tasks to 1.0 to both, loss becomes extreme in a good way)
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --debug_no_resume \
    --lr_scheduler constant \
    --tag test \
    --config_file MetaICL/config/toy.json \
    --data_dir MetaICL/data_toy \
    --debug_fixed_order \
    --samples_per_epoch 100 \
    --lr_embedding 1e-3 \
    --lr_other 1e-3 \
    --eval_pretrained

# now lets check gs padleft and padright with float32, yes they get same gs loss and final eval losses
# also tried commenting out gs part and see if the 2step inference gets same loss as 1step, pad left and pad right
accelerate launch --main_process_port $MASTER_PORT encoder_decoder_noprogram_nlp/train.py \
    --untrainable_nbit 32 \
    --trainable_nbit 32 \
    --no_tf32 \
    --no_flash_attn \
    --debug_no_resume \
    --lr_scheduler constant \
    --tag test \
    --config_file MetaICL/config/toy.json \
    --data_dir MetaICL/data_toy \
    --debug_fixed_order \
    --samples_per_epoch 100 \
    --lr_embedding 1e-3 \
    --lr_other 1e-3 \
    --eval_pretrained \
    --train_gs_lr 10.0 \
    --train_gs_batch_size 100000 \
    --train_gs_optimizer sgd \
    --train_gs_max_grad_norm 1e8 \
    --train_gs_iters 250 \
    --train_gs_take_best \
    --pad_side left

# now lets actually fit properly on a custom dataset
# note llama1b is strong so it looks at demonstration pair and already can answer correctly, so we just care about making loss more extreme
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --debug_no_resume \
    --lr_scheduler constant \
    --tag test \
    --config_file MetaICL/config/toy.json \
    --data_dir MetaICL/data_toy_gs \
    --debug_fixed_order \
    --samples_per_epoch 100 \
    --lr_embedding 1e-3 \
    --lr_other 1e-3 \
    --eval_pretrained \
    --train_gs_lr 1.0 \
    --train_gs_batch_size 100000 \
    --train_gs_optimizer sgd \
    --train_gs_max_grad_norm 1e8 \
    --train_gs_iters 100 \
    --train_gs_take_best
