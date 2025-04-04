# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0213_autoregressive/0213_7_separatecolortokens.sh

# ar separate color tokens
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0214/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_separate_color_tokens \
    --separate_color_tokens \
    --wandb

# Submitted batch job 57265640