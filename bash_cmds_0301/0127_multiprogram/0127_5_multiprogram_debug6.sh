# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0127_multiprogram/0127_5_multiprogram_debug6.sh

# noaug
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder0129_multiprogram/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --eval_batch_size 2 \
    --tag 0127_multiprogram_noaug \
    --no_basic_aug \
    --wandb

# Submitted batch job 56705639