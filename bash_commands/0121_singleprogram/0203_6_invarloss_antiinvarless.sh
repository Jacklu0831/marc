# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0121_singleprogram/0203_6_invarloss_antiinvarless.sh

# single invar1e-3 antiinvarratio0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram/train.py \
    --tag 0203_single_invar1e-3_antiinvarratio0.1 \
    --invar_loss_lambda 1e-3 \
    --anti_invar_ratio 0.1 \
    --wandb

# single invar1e-3 antiinvarratio0.3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram/train.py \
    --tag 0203_single_invar1e-3_antiinvarratio0.3 \
    --invar_loss_lambda 1e-3 \
    --anti_invar_ratio 0.3 \
    --wandb

# single invar1e-3 antiinvarratio0.7
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram/train.py \
    --tag 0203_single_invar1e-3_antiinvarratio0.7 \
    --invar_loss_lambda 1e-3 \
    --anti_invar_ratio 0.7 \
    --wandb

# Submitted batch job 56909678
# Submitted batch job 56909679
# Submitted batch job 56909680