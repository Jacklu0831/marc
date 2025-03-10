# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0310_noprogram/0310_0_base.sh

# noprogram repro
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_0310/train.py \
    --lr_scheduler constant \
    --min_num_pair 8 \
    --tag 0310_noprogram_repro \
    --wandb

# noprogram nobos
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram_0310/train.py \
    --lr_scheduler constant \
    --min_num_pair 8 \
    --tag 0310_noprogram_nobos \
    --no_bos \
    --wandb

# Submitted batch job 58139594
# Submitted batch job 58139595