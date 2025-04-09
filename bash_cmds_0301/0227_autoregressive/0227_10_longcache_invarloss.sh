# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0227_autoregressive/0227_10_longcache_invarloss.sh
# USES NTOKEN4
# with mixedprecisionfix

# arlongcache invar0.1margin0.3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0304/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_invar0.1margin0.3 \
    --invar_loss_lambda 0.1 \
    --invar_loss_margin 0.3 \
    --invar_loss_offset_epochs 4 \
    --invar_loss_linear_epochs 4 \
    --wandb

# arlongcache invar0.1margin0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0304/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_invar0.1margin0.5 \
    --invar_loss_lambda 0.1 \
    --invar_loss_margin 0.5 \
    --invar_loss_offset_epochs 4 \
    --invar_loss_linear_epochs 4 \
    --wandb

# arlongcache invar0.5margin0.3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0304/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_invar0.5margin0.3 \
    --invar_loss_lambda 0.5 \
    --invar_loss_margin 0.3 \
    --invar_loss_offset_epochs 4 \
    --invar_loss_linear_epochs 4 \
    --wandb

# arlongcache invar0.5margin0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0304/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_invar0.5margin0.5 \
    --invar_loss_lambda 0.5 \
    --invar_loss_margin 0.5 \
    --invar_loss_offset_epochs 4 \
    --invar_loss_linear_epochs 4 \
    --wandb

# Submitted batch job 57901771
# Submitted batch job 57901772
# Submitted batch job 57901773
# Submitted batch job 57901774