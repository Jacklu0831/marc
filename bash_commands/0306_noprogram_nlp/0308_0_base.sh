# python make_sbatch.py --ngpu 1 --time 24 --bash_files bash_commands/0306_noprogram_nlp/0308_0_base.sh

# noprogram nlp excludefirst
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0308_noprogram_nlp_excludefirst \
    --loss_type exclude_first \
    --wandb

# noprogram nlp onlylast
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_nlp/train.py \
    --lr_scheduler constant \
    --tag 0308_noprogram_nlp_onlylast \
    --loss_type only_last \
    --wandb

# Submitted batch job 58103466
# Submitted batch job 58103467