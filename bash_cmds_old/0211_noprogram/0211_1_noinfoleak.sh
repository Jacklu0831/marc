# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0211_noprogram/0211_1_noinfoleak.sh

# noprogram noinfoleak
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag 0211_noprogram_noinfoleak \
    --wandb

# Submitted batch job 57330573