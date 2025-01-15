# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0113_more_multigpu/0113_0_base.sh

# lora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_lora \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --wandb

# nolora
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_nolora \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --no_lora \
    --wandb

# noflashattn
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_noflashattn \
    --compact_grids \
    --max_seq_len 5120 \
    --tie_models \
    --wandb
