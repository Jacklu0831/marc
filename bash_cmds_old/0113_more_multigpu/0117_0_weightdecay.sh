# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0113_more_multigpu/0117_0_weightdecay.sh

# weightdecay
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --tag 0113_weightdecay \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --weight_decay 0.01 \
    --wandb

# Submitted batch job 55987757