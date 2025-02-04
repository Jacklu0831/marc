# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0121_singleprogram/0203_1_invarloss_antiinvar.sh

# single invar1e-1 antiinvarratio0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram/train.py \
    --tag 0203_single_invar1e-1_antiinvarratio0.5 \
    --invar_loss_lambda 1e-1 \
    --anti_invar_ratio 0.5 \
    --wandb

# single invar1e-2 antiinvarratio0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram/train.py \
    --tag 0203_single_invar1e-2_antiinvarratio0.5 \
    --invar_loss_lambda 1e-2 \
    --anti_invar_ratio 0.5 \
    --wandb

# single invar1e-3 antiinvarratio0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram/train.py \
    --tag 0203_single_invar1e-3_antiinvarratio0.5 \
    --invar_loss_lambda 1e-3 \
    --anti_invar_ratio 0.5 \
    --wandb

# single invar1e-4 antiinvarratio0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram/train.py \
    --tag 0203_single_invar1e-4_antiinvarratio0.5 \
    --invar_loss_lambda 1e-4 \
    --anti_invar_ratio 0.5 \
    --wandb

# single invar1e-5 antiinvarratio0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram/train.py \
    --tag 0203_single_invar1e-5_antiinvarratio0.5 \
    --invar_loss_lambda 1e-5 \
    --anti_invar_ratio 0.5 \
    --wandb

# Submitted batch job 56865451
# Submitted batch job 56865452
# Submitted batch job 56865453
# Submitted batch job 56865454
# Submitted batch job 56865455