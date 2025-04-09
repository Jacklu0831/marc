# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0121_singleprogram/0203_8_colorequiv.sh

# single nocolorequiv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram_0205/train.py \
    --tag 0203_single_nocolorequiv \
    --color_equiv \
    --wandb

# single colorequiv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram_0205/train.py \
    --tag 0203_single_colorequiv \
    --color_equiv \
    --wandb

# Submitted batch job 56984603
# Submitted batch job 56984604