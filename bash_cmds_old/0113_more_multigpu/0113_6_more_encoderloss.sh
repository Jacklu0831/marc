# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0113_more_multigpu/0113_6_more_encoderloss.sh

# encoderloss2.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_encoderloss2.0 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --encoder_loss_lambda 2.0 \
    --wandb

# encoderloss3.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_encoderloss3.0 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --encoder_loss_lambda 3.0 \
    --wandb

# Submitted batch job 55905940
# Submitted batch job 55905941