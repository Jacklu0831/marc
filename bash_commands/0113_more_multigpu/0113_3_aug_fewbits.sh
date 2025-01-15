# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0113_more_multigpu/0113_3_aug_fewbits.sh

# aug0.0 bit3.6-16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_aug0.0_bit3.6-16 \
    --augment_ratio 0.0 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --untrainable_nbit 3.6 \
    --wandb

# aug0.3 bit3.6-16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_aug0.3_bit3.6-16 \
    --augment_ratio 0.3 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --untrainable_nbit 3.6 \
    --wandb

# aug0.0 bit16-16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_aug0.0_bit16-16 \
    --augment_ratio 0.0 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --untrainable_nbit 16 \
    --wandb

# aug0.3 bit16-16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_aug0.3_bit16-16 \
    --augment_ratio 0.3 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --untrainable_nbit 16 \
    --wandb
