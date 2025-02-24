# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0218_noprogram/0218_0_separatecolor.sh

# noprogram separatecolortoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag 0218_noprogram_separatecolortoken \
    --wandb

# noprogram noseparatecolortoken
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/train.py \
    --lr_scheduler constant \
    --tag 0218_noprogram_noseparatecolortoken \
    --no_separate_color_token \
    --wandb

# Submitted batch job 57429294
# Submitted batch job 57429295