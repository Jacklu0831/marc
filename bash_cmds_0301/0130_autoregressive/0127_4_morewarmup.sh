# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0130_autoregressive/0127_4_morewarmup.sh

# ar novae warmup2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive/train.py \
    --tag 0202_ar_novae_warmup2 \
    --train_no_sample \
    --eval_no_sample \
    --warmup_epoch 2 \
    --wandb

# ar novae warmup4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive/train.py \
    --tag 0202_ar_novae_warmup4 \
    --train_no_sample \
    --eval_no_sample \
    --warmup_epoch 4 \
    --wandb

# Submitted batch job 56968767
# Submitted batch job 56968768