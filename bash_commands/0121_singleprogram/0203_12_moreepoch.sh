# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0121_singleprogram/0203_12_moreepoch.sh

# single epoch30
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram_0206/train.py \
    --tag 0206_epoch30 \
    --num_epochs 30 \
    --wandb

# single epoch35
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram_0206/train.py \
    --tag 0206_epoch35 \
    --num_epochs 35 \
    --wandb

# cancelled for AR runs
# Submitted batch job 57014768
# Submitted batch job 57014769