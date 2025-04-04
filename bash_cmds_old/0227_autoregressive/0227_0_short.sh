# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0227_autoregressive/0227_0_short.sh

# arshort
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arshort \
    --wandb

# arshort evalbs1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arshort_evalbs1 \
    --eval_batch_size 1 \
    --wandb

# arshort concatprograms
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arshort_concatprograms \
    --concat_programs \
    --eval_batch_size 1 \
    --no_residual \
    --wandb

# arshort notf32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0227_arshort_notf32 \
    --no_tf32 \
    --wandb

# Submitted batch job 57840135
# Submitted batch job 57840136
# Submitted batch job 57840199
# Submitted batch job 57840138