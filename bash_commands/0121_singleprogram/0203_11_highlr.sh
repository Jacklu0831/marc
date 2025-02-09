# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0121_singleprogram/0203_11_highlr.sh

# single lr1.5e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram_0206/train.py \
    --tag 0206_lr1.5e-4 \
    --lr_other 1.5e-4 \
    --lr_embedding 1.5e-5 \
    --wandb

# single lr2e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram_0206/train.py \
    --tag 0206_lr2e-4 \
    --lr_other 2e-4 \
    --lr_embedding 2e-5 \
    --wandb

# Submitted batch job 57014766
# Submitted batch job 57014767