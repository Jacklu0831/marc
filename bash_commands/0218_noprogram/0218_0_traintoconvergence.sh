# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0218_noprogram/0218_0_traintoconvergence.sh

# noprogram traintoconvergence
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag 0218_noprogram_traintoconvergence \
    --wandb

# Submitted batch job 57397623