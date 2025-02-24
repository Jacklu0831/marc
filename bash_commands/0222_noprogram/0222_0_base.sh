# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0222_noprogram/0222_0_base.sh

# noprogram base
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag 0222_noprogram_base \
    --wandb

# noprogram ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag 0222_noprogram_ntoken32 \
    --ntokens 32 \
    --wandb

# noprogram ntoken32 cutoff
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag 0222_noprogram_ntoken32_cutoff \
    --ntokens 32 \
    --attention_cutoff \
    --wandb

# noprogram ntoken32 cutoff attendprev
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag 0222_noprogram_ntoken32_cutoff_attendprev \
    --ntokens 32 \
    --attention_cutoff \
    --attend_prev_programs \
    --wandb

# Submitted batch job 57510115
# Submitted batch job 57510116
# Submitted batch job 57510117
# Submitted batch job 57510118