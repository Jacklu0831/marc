# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0130_autoregressive/0207_1_prior_ntokens.sh

# ar prior novae ntoken2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive_0207/train.py \
    --tag 0207_prior_novae_ntoken2 \
    --train_no_sample \
    --eval_no_sample \
    --ntokens 2 \
    --wandb

# ar prior novae ntoken8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive_0207/train.py \
    --tag 0207_prior_novae_ntoken8 \
    --train_no_sample \
    --eval_no_sample \
    --ntokens 8 \
    --wandb

# ar prior novae ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive_0207/train.py \
    --tag 0207_prior_novae_ntoken32 \
    --train_no_sample \
    --eval_no_sample \
    --ntokens 32 \
    --wandb

# ar prior novae ntoken128
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive_0207/train.py \
    --tag 0207_prior_novae_ntoken128 \
    --train_no_sample \
    --eval_no_sample \
    --ntokens 128 \
    --wandb

# cancelled because should just do fixed 8 pairs
# Submitted batch job 57024536
# Submitted batch job 57024537
# Submitted batch job 57024538
# Submitted batch job 57024539