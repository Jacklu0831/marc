# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0113_more_multigpu/0113_7_ntoken.sh

# ntoken1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_ntoken1 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --num_virtual_tokens 1 \
    --wandb

# ntoken4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_ntoken4 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --num_virtual_tokens 4 \
    --wandb

# ntoken16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_ntoken16 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --num_virtual_tokens 16 \
    --wandb

# ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_ntoken32 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --num_virtual_tokens 32 \
    --wandb

# ntoken64
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_ntoken64 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --num_virtual_tokens 64 \
    --wandb

# Submitted batch job 55913694
# Submitted batch job 55913688
# Submitted batch job 55913689
# Submitted batch job 55913690
# Submitted batch job 55987774