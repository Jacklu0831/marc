# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0222_noprogram/0309_1_extraaug.sh

# noprogram extraaug0.3single
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --min_num_pair 8 \
    --tag 0309_noprogram_extraaug0.3single \
    --extra_augment_ratio 0.3 \
    --extra_augment_single_grid \
    --wandb

# noprogram extraaug0.3nosingle
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --min_num_pair 8 \
    --tag 0309_noprogram_extraaug0.3nosingle \
    --extra_augment_ratio 0.3 \
    --wandb

# Submitted batch job 58107179
# Submitted batch job 58107180