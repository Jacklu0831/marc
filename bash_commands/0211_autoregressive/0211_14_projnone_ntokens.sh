# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0211_autoregressive/0211_14_projnone_ntokens.sh

# ar projnone ntokens8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0213/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_projnone_ntokens8 \
    --projection_type none \
    --ntokens 8 \
    --wandb

# ar projnone ntokens16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0213/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_projnone_ntokens16 \
    --projection_type none \
    --ntokens 16 \
    --wandb

# ar projnone ntokens64
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0213/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_projnone_ntokens64 \
    --projection_type none \
    --ntokens 64 \
    --wandb

# ar projnone ntokens128
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0213/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_projnone_ntokens128 \
    --projection_type none \
    --ntokens 128 \
    --wandb

# Submitted batch job 57245026
# Submitted batch job 57245027
# Submitted batch job 57245028
# Submitted batch job 57245029