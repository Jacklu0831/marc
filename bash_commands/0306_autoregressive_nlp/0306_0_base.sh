# python make_sbatch.py --ngpu 1 --time 36 --bash_files bash_commands/0306_autoregressive_nlp/0306_0_base.sh

# arlongcache nlp base
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_nlp/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0306_arlongcache_nlp_base \
    --wandb

# arlongcache nlp excludefirst
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_nlp/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0306_arlongcache_nlp_excludefirst \
    --loss_type exclude_first \
    --wandb

# Submitted batch job 58051430
# Submitted batch job 58051431