# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0317_autoregressive/0317_5_attentionreductionratio.sh

# arlongcache attentionreductionratio0.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_attentionreductionratio0.0 \
    --attention_reduction_ratio 0.0 \
    --wandb

# arlongcache attentionreductionratio0.25
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_attentionreductionratio0.25 \
    --attention_reduction_ratio 0.25 \
    --wandb

# arlongcache attentionreductionratio0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_attentionreductionratio0.5 \
    --attention_reduction_ratio 0.5 \
    --wandb

# arlongcache attentionreductionratio0.75
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_attentionreductionratio0.75 \
    --attention_reduction_ratio 0.75 \
    --wandb

# arlongcache attentionreductionratio0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_attentionreductionratio0.9 \
    --attention_reduction_ratio 0.9 \
    --wandb
