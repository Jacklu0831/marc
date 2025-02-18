# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0213_autoregressive/0213_9_concatprogram.sh

# ar concatprogram noresidual evalbatchsize1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0215/train.py \
    --lr_scheduler constant \
    --tag 0213_ar_concatprogram_noresidual_evalbatchsize1 \
    --concat_programs \
    --eval_batch_size 1 \
    --wandb

# ar concatprogram noresidual evalbatchsize1 norm
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0215/train.py \
    --lr_scheduler constant \
    --tag 0213_ar_concatprogram_noresidual_evalbatchsize1_norm \
    --concat_programs \
    --eval_batch_size 1 \
    --normalize \
    --wandb

# Submitted batch job 57319922
# Submitted batch job 57319923