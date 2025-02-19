# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0211_singleprogram/0211_1_ntokens.sh

# single ntokens16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram/train.py \
    --tie_models \
    --lr_scheduler constant \
    --tag 0211_single_ntokens16 \
    --ntokens 16 \
    --wandb

# single ntokens32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram/train.py \
    --tie_models \
    --lr_scheduler constant \
    --tag 0211_single_ntokens32 \
    --ntokens 32 \
    --wandb

# single ntokens128
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram/train.py \
    --tie_models \
    --lr_scheduler constant \
    --tag 0211_single_ntokens128 \
    --ntokens 128 \
    --wandb

# Submitted batch job 57149777
# Submitted batch job 57149778
# Submitted batch job 57149779