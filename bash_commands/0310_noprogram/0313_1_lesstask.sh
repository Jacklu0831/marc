# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0310_noprogram/0313_1_lesstask.sh

# noprogram 100task
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_0313/train.py \
    --lr_scheduler constant \
    --tag 0313_noprogram_100task \
    --train_data_dir ./data/re-arc/train_data_100/tasks \
    --wandb

# noprogram 200task
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_0313/train.py \
    --lr_scheduler constant \
    --tag 0313_noprogram_200task \
    --train_data_dir ./data/re-arc/train_data_200/tasks \
    --wandb

# Submitted batch job 58265106
# Submitted batch job 58265107