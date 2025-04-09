# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0218_autoregressive/0218_18_desperate.sh

# ar repro notokenweighted
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0223/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_repro_notokenweighted \
    --wandb

# ar float16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0223/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0218_ar_float16 \
    --wandb

# ar trainpadleft
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0223/train.py \
    --lr_scheduler constant \
    --token_weighted_loss \
    --tag 0218_ar_trainpadleft \
    --train_pad_side left \
    --wandb

# Submitted batch job 57581804
# Submitted batch job 57577164
# Submitted batch job 57577166