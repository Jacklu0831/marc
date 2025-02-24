# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0217_autoregressive/0217_3_gradaccum.sh

# ar gradaccum4 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217_eval/train.py \
    --lr_scheduler constant \
    --tag 0213_ar_gradaccum4_lr2e-4 \
    --grad_accum_steps 4 \
    --lr_embedding 1e-5 \
    --lr_program 1e-4 \
    --lr_prior 1e-4 \
    --lr_other 1e-4 \
    --wandb

# ar gradaccum8 lr3e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217_eval/train.py \
    --lr_scheduler constant \
    --tag 0213_ar_gradaccum8_lr3e-4 \
    --grad_accum_steps 8 \
    --lr_embedding 3e-5 \
    --lr_program 3e-4 \
    --lr_prior 3e-4 \
    --lr_other 3e-4 \
    --wandb

# ar gradaccum8 lr4e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217_eval/train.py \
    --lr_scheduler constant \
    --tag 0213_ar_gradaccum8_lr4e-4 \
    --grad_accum_steps 8 \
    --lr_embedding 4e-5 \
    --lr_program 4e-4 \
    --lr_prior 4e-4 \
    --lr_other 4e-4 \
    --wandb

# ar gradaccum16 lr4e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217_eval/train.py \
    --lr_scheduler constant \
    --tag 0213_ar_gradaccum16_lr4e-4 \
    --grad_accum_steps 16 \
    --lr_embedding 4e-5 \
    --lr_program 4e-4 \
    --lr_prior 4e-4 \
    --lr_other 4e-4 \
    --wandb

# ar gradaccum16 lr6e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217_eval/train.py \
    --lr_scheduler constant \
    --tag 0213_ar_gradaccum16_lr6e-4 \
    --grad_accum_steps 16 \
    --lr_embedding 6e-5 \
    --lr_program 6e-4 \
    --lr_prior 6e-4 \
    --lr_other 6e-4 \
    --wandb

# ar gradaccum16 lr8e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_autoregressive_0217_eval/train.py \
    --lr_scheduler constant \
    --tag 0213_ar_gradaccum16_lr8e-4 \
    --grad_accum_steps 16 \
    --lr_embedding 8e-5 \
    --lr_program 8e-4 \
    --lr_prior 8e-4 \
    --lr_other 8e-4 \
    --wandb

# Submitted batch job 57338621
# Submitted batch job 57338622
# Submitted batch job 57338623
# Submitted batch job 57338624
# Submitted batch job 57338625
# Submitted batch job 57338626