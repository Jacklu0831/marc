# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0121_singleprogram/0203_13_tiemodels.sh

# single tiemodels
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram_0206/train.py \
    --tag 0206_tiemodels \
    --tie_models \
    --wandb

# Submitted batch job 57015357