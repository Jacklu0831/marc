# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0211_singleprogram/0211_0_base.sh

# single base
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram/train.py \
    --tie_models \
    --lr_scheduler constant \
    --tag 0211_single_base \
    --wandb

# single nobasicaug
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram/train.py \
    --tie_models \
    --lr_scheduler constant \
    --tag 0211_single_nobasicaug \
    --no_color_permute \
    --no_pair_permute \
    --no_d8 \
    --wandb

# Submitted batch job 57149751
# Submitted batch job 57149752 # ideky killed