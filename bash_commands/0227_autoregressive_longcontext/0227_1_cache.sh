# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0227_autoregressive_longcontext/0227_1_cache.sh
# 40 epochs in ~60 hours on 2gpu, ~80 hours notf32

# arlongcache
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache \
    --wandb

# arlongcache demondropout0.03
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_demondropout0.03 \
    --demonstration_dropout 0.03 \
    --wandb

# arlongcache demondropout0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_demondropout0.1 \
    --demonstration_dropout 0.1 \
    --wandb

# arlongcache demondropout0.3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_demondropout0.3 \
    --demonstration_dropout 0.3 \
    --wandb

# arlongcache notf32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_notf32 \
    --no_tf32 \
    --wandb

# arlongcache evalbs1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_evalbs1 \
    --eval_batch_size 1 \
    --wandb

# # TEST MEMORY
# accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
#     --lr_scheduler constant \
#     --token_weighted_loss \
#     --tag test \
#     --log_every 1 \
#     --debug_len 8192 \
#     --samples_per_epoch 256

# Submitted batch job 57692151 # zhenbang
# Submitted batch job 57692167 # mengye
# Submitted batch job 57692168
# Submitted batch job 57692169
# Submitted batch job 57692705 # mine
# Submitted batch job 57692707