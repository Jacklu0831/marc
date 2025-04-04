# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0218_autoregressive/0218_4_concatprograms.sh

# ar concatprogram
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_concatprogram \
    --concat_programs \
    --no_residual \
    --eval_batch_size 1 \
    --wandb

# ar concatprogram nonorm
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_concatprogram_nonorm \
    --concat_programs \
    --no_residual \
    --eval_batch_size 1 \
    --no_normalize \
    --wandb

# Submitted batch job 57364994
# Submitted batch job 57364995