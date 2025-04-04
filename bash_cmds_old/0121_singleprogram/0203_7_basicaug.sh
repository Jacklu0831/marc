# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0121_singleprogram/0203_7_basicaug.sh

# single basicaug
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram_0205/train.py \
    --tag 0203_single_basicaug \
    --basic_aug \
    --wandb

# Submitted batch job 56990502