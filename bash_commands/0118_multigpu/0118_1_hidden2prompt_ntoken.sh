# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0118_multigpu/0118_1_hidden2prompt_ntoken.sh

# hidden2prompt ntoken1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new6/train.py \
    --tag 0118_hidden2prompt_ntoken1 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --conditioning_method hidden2prompt \
    --num_virtual_tokens 1 \
    --wandb

# hidden2prompt ntoken4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new6/train.py \
    --tag 0118_hidden2prompt_ntoken4 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --conditioning_method hidden2prompt \
    --num_virtual_tokens 4 \
    --wandb

# hidden2prompt ntoken16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new6/train.py \
    --tag 0118_hidden2prompt_ntoken16 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --conditioning_method hidden2prompt \
    --num_virtual_tokens 16 \
    --wandb

# hidden2prompt ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new6/train.py \
    --tag 0118_hidden2prompt_ntoken32 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --conditioning_method hidden2prompt \
    --num_virtual_tokens 32 \
    --wandb

# hidden2prompt ntoken64
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new6/train.py \
    --tag 0118_hidden2prompt_ntoken64 \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --conditioning_method hidden2prompt \
    --num_virtual_tokens 64 \
    --wandb

# Submitted batch job 56021539
# Submitted batch job 56021540
# Submitted batch job 56021541
# Submitted batch job 56021542
# Submitted batch job 56021543