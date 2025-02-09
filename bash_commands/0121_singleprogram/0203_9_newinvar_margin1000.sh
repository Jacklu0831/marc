# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0121_singleprogram/0203_9_newinvar_margin1000.sh

# single invar1e-4 antiinvarratio0.5 margin1000
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram_0206/train.py \
    --tag 0206_single_invar1e-4_antiinvarratio0.5_margin1000 \
    --invar_loss_lambda 1e-4 \
    --anti_invar_ratio 0.5 \
    --anti_invar_margin 1000 \
    --wandb

# single invar1e-5 antiinvarratio0.5 margin1000
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram_0206/train.py \
    --tag 0206_single_invar1e-5_antiinvarratio0.5_margin1000 \
    --invar_loss_lambda 1e-5 \
    --anti_invar_ratio 0.5 \
    --anti_invar_margin 1000 \
    --wandb

# single invar1e-6 antiinvarratio0.5 margin1000
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram_0206/train.py \
    --tag 0206_single_invar1e-6_antiinvarratio0.5_margin1000 \
    --invar_loss_lambda 1e-6 \
    --anti_invar_ratio 0.5 \
    --anti_invar_margin 1000 \
    --wandb

# single invar1e-7 antiinvarratio0.5 margin1000
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_singleprogram_0206/train.py \
    --tag 0206_single_invar1e-7_antiinvarratio0.5_margin1000 \
    --invar_loss_lambda 1e-7 \
    --anti_invar_ratio 0.5 \
    --anti_invar_margin 1000 \
    --wandb

# Submitted batch job 56997371
# Submitted batch job 56997372
# Submitted batch job 56997373
# Submitted batch job 56997374