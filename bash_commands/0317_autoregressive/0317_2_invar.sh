# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0317_autoregressive/0317_2_invar.sh

# arlongcache invar0.03margin0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_invar0.03margin0.5 \
    --invar_loss_lambda 0.03 \
    --invar_loss_margin 0.5 \
    --wandb

# arlongcache invar0.1margin0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_invar0.1margin0.5 \
    --invar_loss_lambda 0.1 \
    --invar_loss_margin 0.5 \
    --wandb

# arlongcache invar0.3margin0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_invar0.3margin0.5 \
    --invar_loss_lambda 0.3 \
    --invar_loss_margin 0.5 \
    --wandb

# arlongcache invar1.0margin0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --no_bos \
    --token_weighted_loss \
    --tag 0317_arlongcache_invar1.0margin0.5 \
    --invar_loss_lambda 1.0 \
    --invar_loss_margin 0.5 \
    --wandb
