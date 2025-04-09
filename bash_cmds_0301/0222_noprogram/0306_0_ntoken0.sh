# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0222_noprogram/0306_0_ntoken0.sh

# noprogram ntoken0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_0303/train.py \
    --lr_scheduler constant \
    --min_num_pair 8 \
    --tag 0306_noprogram_ntoken0 \
    --ntokens 0 \
    --wandb

# Submitted batch job 58043873