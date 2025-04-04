# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0127_multiprogram/0127_4_multiprogram_lowencoderloss.sh

# encoderloss0.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder0125_multiprogram/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0127_multiprogram_encoderloss0.0 \
    --encoder_loss_lambda 0.0 \
    --wandb

# encoderloss0.25
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder0125_multiprogram/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0127_multiprogram_encoderloss0.25 \
    --encoder_loss_lambda 0.25 \
    --wandb

# encoderloss0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder0125_multiprogram/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0127_multiprogram_encoderloss0.5 \
    --encoder_loss_lambda 0.5 \
    --wandb

# encoderloss0.75
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder0125_multiprogram/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0127_multiprogram_encoderloss0.75 \
    --encoder_loss_lambda 0.75 \
    --wandb
