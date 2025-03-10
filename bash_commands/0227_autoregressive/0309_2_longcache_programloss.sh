# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0227_autoregressive/0309_2_longcache_programloss.sh

# arlongcache programloss0.03random
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0309_arlongcache_programloss0.03random \
    --program_type random \
    --program_loss_lambda 0.03 \
    --wandb

# arlongcache programloss0.1random
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0309_arlongcache_programloss0.1random \
    --program_type random \
    --program_loss_lambda 0.1 \
    --wandb

# arlongcache programloss0.3random
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0309_arlongcache_programloss0.3random \
    --program_type random \
    --program_loss_lambda 0.3 \
    --wandb

# arlongcache programloss1.0random
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0309_arlongcache_programloss1.0random \
    --program_type random \
    --program_loss_lambda 1.0 \
    --wandb

# Submitted batch job 58103985
# Submitted batch job 58103986
# Submitted batch job 58103987
# Submitted batch job 58103988