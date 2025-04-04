# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0127_multiprogram/0127_7_multiprogram_subsetinvar.sh

# subsetinvar1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder0129_multiprogram/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0127_multiprogram_subsetinvar1e-4 \
    --subset_invar_loss_lambda 1e-4 \
    --wandb

# subsetinvar1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder0129_multiprogram/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0127_multiprogram_subsetinvar1e-3 \
    --subset_invar_loss_lambda 1e-3 \
    --wandb

# subsetinvar1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder0129_multiprogram/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0127_multiprogram_subsetinvar1e-2 \
    --subset_invar_loss_lambda 1e-2 \
    --wandb

# subsetinvar1e-1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder0129_multiprogram/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0127_multiprogram_subsetinvar1e-1 \
    --subset_invar_loss_lambda 1e-1 \
    --wandb
