# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0127_multiprogram/0204_1_limiteval_colon.sh

# multi limit eval
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_multiprogram_0203/train.py \
    --tag 0203_multi_limiteval \
    --limit_eval_to_max_program \
    --wandb

# multi colon encoding
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_multiprogram_0203/train.py \
    --tag 0203_multi_colon \
    --colon_encoding \
    --wandb

# Submitted batch job 56945058
# Submitted batch job 56945059