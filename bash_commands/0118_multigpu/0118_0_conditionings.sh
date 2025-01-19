# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0118_multigpu/0118_0_conditionings.sh

# prefix2prefix
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new6/train.py \
    --tag 0118_prefix2prefix \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --conditioning_method prefix2prefix \
    --wandb

# hidden2prompt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new6/train.py \
    --tag 0118_hidden2prompt \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --conditioning_method hidden2prompt \
    --wandb

# Submitted batch job 56021532
# Submitted batch job 56021535