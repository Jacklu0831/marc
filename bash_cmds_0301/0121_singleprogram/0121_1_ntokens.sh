# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0121_singleprogram/0121_1_ntokens.sh

# ntokens128
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder7/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0121_ntokens128 \
    --ntokens 128 \
    --wandb

# ntokens256
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder7/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0121_ntokens256 \
    --ntokens 256 \
    --wandb

# ntokens512
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder7/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0121_ntokens512 \
    --ntokens 512 \
    --wandb

# Submitted batch job 56205684
# Submitted batch job 56205685
# Submitted batch job 56205686 # missing??