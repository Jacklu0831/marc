# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0310_autoregressive/0313_2_attentionreduction.sh

# arlongcache attentionreduction0.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0313_arlongcache_attentionreduction0.0 \
    --attention_reduction_ratio 0.0 \
    --wandb

# arlongcache attentionreduction0.25
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0313_arlongcache_attentionreduction0.25 \
    --attention_reduction_ratio 0.25 \
    --wandb

# arlongcache attentionreduction0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0313_arlongcache_attentionreduction0.5 \
    --attention_reduction_ratio 0.5 \
    --wandb

# arlongcache attentionreduction0.75
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0313_arlongcache_attentionreduction0.75 \
    --attention_reduction_ratio 0.75 \
    --wandb

# arlongcache attentionreduction0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0313_arlongcache_attentionreduction0.9 \
    --attention_reduction_ratio 0.9 \
    --wandb

# arlongcache attentionreduction1.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0313_arlongcache_attentionreduction1.0 \
    --attention_reduction_ratio 1.0 \
    --wandb

# Submitted batch job 58427384
# Submitted batch job 58427385
# Submitted batch job 58427386
# Submitted batch job 58427387
# Submitted batch job 58427388
# Submitted batch job 58427389