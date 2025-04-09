# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0127_multiprogram/0204_0_base.sh

# multi base
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_multiprogram/train.py \
    --tag 0203_multi_base \
    --wandb

# multi voting
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_multiprogram/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --tag 0203_multi_voting \
    --wandb

# Submitted batch job 56865727
# Submitted batch job 56865728