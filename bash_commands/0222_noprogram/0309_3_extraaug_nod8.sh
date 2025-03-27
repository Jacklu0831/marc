# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0222_noprogram/0309_3_extraaug_nod8.sh

# noprogram extraaug0.3single nod8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --min_num_pair 8 \
    --tag 0309_noprogram_extraaug0.3single_nod8 \
    --extra_augment_ratio 0.3 \
    --extra_augment_single_grid \
    --no_d8 \
    --wandb

# noprogram extraaug0.3nosingle nod8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --min_num_pair 8 \
    --tag 0309_noprogram_extraaug0.3nosingle_nod8 \
    --extra_augment_ratio 0.3 \
    --no_d8 \
    --wandb

# noprogram extraaug0.3single nod8 noothers
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --min_num_pair 8 \
    --tag 0309_noprogram_extraaug0.3single_nod8_noothers \
    --extra_augment_ratio 0.3 \
    --extra_augment_single_grid \
    --no_d8 \
    --no_color_permute \
    --no_pair_permute \
    --wandb

# noprogram extraaug0.3nosingle nod8 noothers
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --min_num_pair 8 \
    --tag 0309_noprogram_extraaug0.3nosingle_nod8_noothers \
    --extra_augment_ratio 0.3 \
    --no_d8 \
    --no_color_permute \
    --no_pair_permute \
    --wandb

# Submitted batch job 58181802
# Submitted batch job 58181803
# Submitted batch job 58181804
# Submitted batch job 58181805