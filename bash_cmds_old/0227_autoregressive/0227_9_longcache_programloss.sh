# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0227_autoregressive/0227_9_longcache_programloss.sh
# USES NTOKEN4
# with mixedprecisionfix

# arlongcache programloss0.3concat
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0303/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_programloss0.3concat \
    --program_type concat \
    --program_loss_lambda 0.3 \
    --wandb

# arlongcache programloss1.0concat
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0303/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_programloss1.0concat \
    --program_type concat \
    --program_loss_lambda 1.0 \
    --wandb

# arlongcache programloss0.3random
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0303/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_programloss0.3random \
    --program_type random \
    --program_loss_lambda 0.3 \
    --wandb

# arlongcache programloss1.0random
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_longcontext_caching_0303/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arlongcache_programloss1.0random \
    --program_type random \
    --program_loss_lambda 1.0 \
    --wandb

# Submitted batch job 57867405
# Submitted batch job 57867406
# Submitted batch job 57867407
# Submitted batch job 57867408