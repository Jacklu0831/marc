# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0130_autoregressive/0208_0_prior_accgradaccum.sh

# ar prior novae accgradaccum4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive_0208/train.py \
    --tag 0207_prior_novae_accgradaccum4 \
    --train_no_sample \
    --eval_no_sample \
    --grad_accum_steps 4 \
    --wandb

# ar prior novae accgradaccum8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive_0208/train.py \
    --tag 0207_prior_novae_accgradaccum8 \
    --train_no_sample \
    --eval_no_sample \
    --grad_accum_steps 8 \
    --wandb

# ar prior novae accgradaccum8 lr1.5e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive_0208/train.py \
    --tag 0207_prior_novae_accgradaccum8_lr1.5e-4 \
    --train_no_sample \
    --eval_no_sample \
    --grad_accum_steps 8 \
    --lr_embedding 1.5e-5 \
    --lr_program 1.5e-5 \
    --lr_prior 1.5e-5 \
    --lr_other 1.5e-4 \
    --wandb

# ar prior novae accgradaccum8 lr2e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive_0208/train.py \
    --tag 0207_prior_novae_accgradaccum8_lr2e-4 \
    --train_no_sample \
    --eval_no_sample \
    --grad_accum_steps 8 \
    --lr_embedding 2e-5 \
    --lr_program 2e-5 \
    --lr_prior 2e-5 \
    --lr_other 2e-4 \
    --wandb
