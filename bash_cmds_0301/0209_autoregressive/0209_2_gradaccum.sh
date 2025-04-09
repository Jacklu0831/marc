# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0209_autoregressive/0209_2_gradaccum.sh

# ar gradaccum8 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --tag 0209_ar_gradaccum8_lr1e-4 \
    --train_no_sample \
    --eval_no_sample \
    --grad_accum_steps 8 \
    --lr_embedding 1e-4 \
    --lr_program 1e-4 \
    --lr_prior 1e-4 \
    --lr_other 1e-4 \
    --wandb

# ar gradaccum8 lr2e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --tag 0209_ar_gradaccum8_lr2e-4 \
    --train_no_sample \
    --eval_no_sample \
    --grad_accum_steps 8 \
    --lr_embedding 2e-4 \
    --lr_program 2e-4 \
    --lr_prior 2e-4 \
    --lr_other 2e-4 \
    --wandb

# ar gradaccum16 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --tag 0209_ar_gradaccum16_lr1e-4 \
    --train_no_sample \
    --eval_no_sample \
    --grad_accum_steps 16 \
    --lr_embedding 1e-4 \
    --lr_program 1e-4 \
    --lr_prior 1e-4 \
    --lr_other 1e-4 \
    --wandb

# ar gradaccum16 lr2e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --tag 0209_ar_gradaccum16_lr2e-4 \
    --train_no_sample \
    --eval_no_sample \
    --grad_accum_steps 16 \
    --lr_embedding 2e-4 \
    --lr_program 2e-4 \
    --lr_prior 2e-4 \
    --lr_other 2e-4 \
    --wandb

# ar gradaccum16 lr4e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive/train.py \
    --tag 0209_ar_gradaccum16_lr4e-4 \
    --train_no_sample \
    --eval_no_sample \
    --grad_accum_steps 16 \
    --lr_embedding 4e-4 \
    --lr_program 4e-4 \
    --lr_prior 4e-4 \
    --lr_other 4e-4 \
    --wandb

# Submitted batch job 57077651
# Submitted batch job 57077652
# Submitted batch job 57077653
# Submitted batch job 57077654
# Submitted batch job 57077655 # cancelled cuz loss jump