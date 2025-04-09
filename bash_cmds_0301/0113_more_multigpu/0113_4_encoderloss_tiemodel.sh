# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0113_more_multigpu/0113_4_encoderloss_tiemodel.sh

# encoderloss0.0 tiemodel
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_encoderloss0.0_tiemodel \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --encoder_loss_lambda 0.0 \
    --tie_models \
    --wandb

# encoderloss1.0 tiemodel
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_encoderloss1.0_tiemodel \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --encoder_loss_lambda 1.0 \
    --tie_models \
    --wandb

# encoderloss0.0 notiemodel
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_encoderloss0.0_notiemodel \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --encoder_loss_lambda 0.0 \
    --wandb

# encoderloss1.0 notiemodel
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_encoderloss1.0_notiemodel \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --encoder_loss_lambda 1.0 \
    --wandb

# Submitted batch job 55905929
# Submitted batch job 55905930
# Submitted batch job 55905931
# Submitted batch job 55905932