# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0209_autoregressive/0209_3_tokenization.sh

# ar nodim
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --tag 0211_ar_nodim \
    --train_no_sample \
    --eval_no_sample \
    --no_dim \
    --wandb

# ar separatecolortokens
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --tag 0211_ar_separatecolortokens \
    --train_no_sample \
    --eval_no_sample \
    --separate_color_tokens \
    --wandb

# ar nodim separatecolortokens
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --tag 0211_ar_nodim_separatecolortokens \
    --train_no_sample \
    --eval_no_sample \
    --no_dim \
    --separate_color_tokens \
    --wandb

# died early in greene calamity
# Submitted batch job 57077658
# Submitted batch job 57077659
# Submitted batch job 57077660