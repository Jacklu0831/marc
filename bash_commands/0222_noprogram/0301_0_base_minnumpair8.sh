# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0222_noprogram/0301_0_base_minnumpair8.sh

# noprogram base minnumpair8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag 0222_noprogram_base_minnumpair8 \
    --min_num_pair 8 \
    --wandb

# noprogram ntoken16 minnumpair8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag 0222_noprogram_ntoken16_minnumpair8 \
    --ntokens 16 \
    --min_num_pair 8 \
    --wandb

# noprogram ntoken16 cutoff minnumpair8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag 0222_noprogram_ntoken16_cutoff_minnumpair8 \
    --ntokens 16 \
    --attention_cutoff \
    --min_num_pair 8 \
    --wandb

# noprogram ntoken16 cutoff attendprev minnumpair8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag 0222_noprogram_ntoken16_cutoff_attendprev_minnumpair8 \
    --ntokens 16 \
    --attention_cutoff \
    --attend_prev_programs \
    --min_num_pair 8 \
    --wandb

# Submitted batch job 57840166
# Submitted batch job 57840167
# Submitted batch job 57840168
# Submitted batch job 57840169