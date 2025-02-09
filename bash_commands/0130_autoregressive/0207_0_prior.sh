# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0130_autoregressive/0207_0_prior.sh

# ar prior novae
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive_0207/train.py \
    --tag 0207_prior_novae \
    --train_no_sample \
    --eval_no_sample \
    --wandb

# ar prior novae minnumpair8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive_0207/train.py \
    --tag 0207_prior_novae_minnumpair8 \
    --train_no_sample \
    --eval_no_sample \
    --min_num_pair 8 \
    --wandb

# ar prior novae uniformlr
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder1031_autoregressive_0207/train.py \
    --tag 0207_prior_novae_uniformlr \
    --train_no_sample \
    --eval_no_sample \
    --lr_embedding 1e-4 \
    --lr_program 1e-4 \
    --lr_prior 1e-4 \
    --lr_other 1e-4 \
    --wandb

# Submitted batch job 57024540
# Submitted batch job 57024541
# Submitted batch job 57024542 (cancelled)