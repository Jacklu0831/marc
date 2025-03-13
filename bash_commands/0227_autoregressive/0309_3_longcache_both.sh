# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0227_autoregressive/0309_3_longcache_both.sh

# arlongcache invar0.1margin0.5 programloss0.3random
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0309_arlongcache_invar0.1margin0.5_programloss0.3random \
    --program_type random \
    --program_loss_lambda 0.3 \
    --invar_loss_lambda 0.1 \
    --invar_loss_margin 0.5 \
    --wandb

# arlongcache invar0.1margin0.5 programloss1.0random
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0309_arlongcache_invar0.1margin0.5_programloss1.0random \
    --program_type random \
    --program_loss_lambda 1.0 \
    --invar_loss_lambda 0.1 \
    --invar_loss_margin 0.5 \
    --wandb

# Submitted batch job 58263633
# Submitted batch job 58263634