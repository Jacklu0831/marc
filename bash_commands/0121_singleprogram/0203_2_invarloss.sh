# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0121_singleprogram/0203_2_invarloss.sh

# single invar1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram/train.py \
    --tag 0203_single_invar1e-3 \
    --invar_loss_lambda 1e-3 \
    --wandb

# Submitted batch job 56865456