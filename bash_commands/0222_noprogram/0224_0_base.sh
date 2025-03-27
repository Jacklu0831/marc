# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0222_noprogram/0224_0_base.sh

# noprogram padleft
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag 0224_noprogram_padleft \
    --wandb

# noprogram padleft ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag 0224_noprogram_padleft_ntoken32 \
    --ntokens 32 \
    --wandb

# noprogram padleft ntoken32 cutoff
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag 0224_noprogram_padleft_ntoken32_cutoff \
    --ntokens 32 \
    --attention_cutoff \
    --wandb

# noprogram padleft ntoken32 cutoff attendprev
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag 0224_noprogram_padleft_ntoken32_cutoff_attendprev \
    --ntokens 32 \
    --attention_cutoff \
    --attend_prev_programs \
    --wandb

# Submitted batch job 57580892
# Submitted batch job 57891043
# Submitted batch job 57580894
# Submitted batch job 57580895