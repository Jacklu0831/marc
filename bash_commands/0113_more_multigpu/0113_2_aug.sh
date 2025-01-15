# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0113_more_multigpu/0113_2_aug.sh

# aug0.1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_aug0.1 \
    --augment_ratio 0.1 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --wandb

# aug0.3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_aug0.3 \
    --augment_ratio 0.3 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --wandb

# aug0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_aug0.5 \
    --augment_ratio 0.5 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --wandb

# Submitted batch job 55905922
# Submitted batch job 55905923
# Submitted batch job 55905924