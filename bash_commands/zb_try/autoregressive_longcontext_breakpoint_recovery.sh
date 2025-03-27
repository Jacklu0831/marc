# new embedding baseline single program 
accelerate launch --main_process_port 55126 --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_demondropout0.03 \
    --demonstration_dropout 0.03 \
    --tag autoregressive_longcontext_caching_breakpoint_recovery --wandb\
    --resume\
    --resume_debug
