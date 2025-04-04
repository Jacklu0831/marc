# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0218_autoregressive/0218_15_extraugs.sh

# ar extraaug0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0221/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_extraaug0.1 \
    --extra_augment_ratio 0.1 \
    --wandb

# ar extraaug0.3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0221/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_extraaug0.3 \
    --extra_augment_ratio 0.3 \
    --wandb

# ar extraaug0.1 singlegrid
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0221/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_extraaug0.1_singlegrid \
    --extra_augment_ratio 0.1 \
    --extra_augment_single_grid \
    --wandb

# ar extraaug0.3 singlegrid
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0221/train.py \
    --lr_scheduler constant \
    --tag 0218_ar_extraaug0.3_singlegrid \
    --extra_augment_ratio 0.3 \
    --extra_augment_single_grid \
    --wandb

# Submitted batch job 57485962
# Submitted batch job 57485963
# Submitted batch job 57485964
# Submitted batch job 57485965