# python make_sbatch.py --ngpu 2 --time 24 --bash_files bash_commands/0306_autoregressive_nlp/0310_0_base.sh --burst

# arlongcache nlp base
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_nlp/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --no_bos \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --tag 0310_arlongcache_nlp_base \
    --wandb

# arlongcache nlp excludefirst
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_nlp/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --no_bos \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --tag 0310_arlongcache_nlp_excludefirst \
    --loss_type exclude_first \
    --wandb

# arlongcache nlp delimiteranswer
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_nlp/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --no_bos \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --tag 0310_arlongcache_nlp_delimiteranswer \
    --delimiter " Answer: " \
    --wandb
