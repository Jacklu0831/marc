# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0211_autoregressive/0211_2_emblr.sh

# ar programlr1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_ar_programlr1e-5 \
    --lr_program 1e-5 \
    --lr_prior 1e-5 \
    --wandb

# ar emblr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_ar_emblr1e-4 \
    --lr_embedding 1e-4 \
    --wandb

# ar emblr1e-6
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_ar_emblr1e-6 \
    --lr_embedding 1e-6 \
    --wandb

# ar emblr0.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --train_no_sample \
    --eval_no_sample \
    --residual \
    --lr_scheduler constant \
    --tag 0211_ar_emblr0.0 \
    --lr_embedding 0.0 \
    --wandb

# Submitted batch job 57150076
# Submitted batch job 57150077
# Submitted batch job 57150078
# Submitted batch job 57150079