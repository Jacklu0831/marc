# python make_sbatch.py --ngpu 1 --time 36 --bash_files bash_commands/0306_autoregressive_nlp/0308_0_base.sh

# arlongcache nlp excludefirst
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_nlp/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0308_arlongcache_nlp_excludefirst \
    --loss_type exclude_first \
    --wandb

# arlongcache nlp onlylast
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_nlp/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0308_arlongcache_nlp_onlylast \
    --loss_type only_last \
    --wandb

# Submitted batch job 58103378
# Submitted batch job 58103379