# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0213_autoregressive/0213_0_base.sh

# ar base
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_base \
    --wandb

# ar gradaccum8 lr2e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_gradaccum8_lr2e-4 \
    --grad_accum_steps 8 \
    --lr_embedding 2e-5 \
    --lr_program 2e-4 \
    --lr_prior 2e-4 \
    --lr_other 2e-4 \
    --wandb

# ar norm
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_norm \
    --normalize \
    --wandb

# ar programlr1e-5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_programlr1e-5 \
    --lr_program 1e-5 \
    --lr_prior 1e-5 \
    --wandb

# ar proj shared
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --residual \
    --lr_scheduler constant \
    --tag 0213_ar_projshared \
    --projection_type shared \
    --wandb

# Submitted batch job 57246030
# Submitted batch job 57246031
# Submitted batch job 57246032
# Submitted batch job 57246033
# Submitted batch job 57246034