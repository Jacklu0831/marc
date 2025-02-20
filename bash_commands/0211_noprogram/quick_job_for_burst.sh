# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0211_noprogram/quick_job_for_burst.sh --burst

# noprogram base
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag 0211_noprogram_burst \
    --wandb

# Submitted batch job 33211