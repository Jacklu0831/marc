# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0227_autoregressive/0309_1_longcache_invarloss.sh

# arlongcache invar0.03margin0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0309_arlongcache_invar0.03margin0.5 \
    --invar_loss_lambda 0.03 \
    --invar_loss_margin 0.5 \
    --wandb

# arlongcache invar0.1margin0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0309_arlongcache_invar0.1margin0.5 \
    --invar_loss_lambda 0.1 \
    --invar_loss_margin 0.5 \
    --wandb

# arlongcache invar0.3margin0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0309_arlongcache_invar0.3margin0.5 \
    --invar_loss_lambda 0.3 \
    --invar_loss_margin 0.5 \
    --wandb

# arlongcache invar1.0margin0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0309_arlongcache_invar1.0margin0.5 \
    --invar_loss_lambda 1.0 \
    --invar_loss_margin 0.5 \
    --wandb

# Submitted batch job 58103979
# Submitted batch job 58103980
# Submitted batch job 58103981
# Submitted batch job 58103982

# continue after the 48 hrs
# Submitted batch job 58264451
# Submitted batch job 58264452
# Submitted batch job 58264453
# Submitted batch job 58264454