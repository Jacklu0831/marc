# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0213_autoregressive/0213_9_concatprogram2ndtry.sh

# ar concatprogram noresidual evalbatchsize1 norm ntokens16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0215/train.py \
    --lr_scheduler constant \
    --tag 0213_ar_concatprogram_noresidual_evalbatchsize1_norm_ntokens16 \
    --concat_programs \
    --eval_batch_size 1 \
    --normalize \
    --ntokens 16 \
    --wandb

# ar concatprogram noresidual evalbatchsize1 norm ntokens64
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0215/train.py \
    --lr_scheduler constant \
    --tag 0213_ar_concatprogram_noresidual_evalbatchsize1_norm_ntokens64 \
    --concat_programs \
    --eval_batch_size 1 \
    --normalize \
    --ntokens 64 \
    --wandb

# Submitted batch job 57355394
# Submitted batch job 57355395