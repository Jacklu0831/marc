# python make_sbatch.py --ngpu 1 --time 24 --bash_files bash_commands/0306_noprogram_nlp/0306_0_base.sh

# noprogram nlp base
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0306_noprogram_nlp_base \
    --wandb

# noprogram nlp excludefirst
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0306_noprogram_nlp_excludefirst\
    --loss_type exclude_first \
    --wandb

# Submitted batch job 58051432
# Submitted batch job 58051433
