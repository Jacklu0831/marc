# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0127_multiprogram/0127_6_multiprogram_batchinvar_antiinvar.sh

# batchinvar1e-4 antiinvar0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder0129_multiprogram/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0127_multiprogram_batchinvar1e-4_antiinvar0.5 \
    --batch_invar_loss_lambda 1e-4 \
    --anti_invar_ratio 0.5 \
    --wandb

# batchinvar1e-3 antiinvar0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder0129_multiprogram/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0127_multiprogram_batchinvar1e-3_antiinvar0.5 \
    --batch_invar_loss_lambda 1e-3 \
    --anti_invar_ratio 0.5 \
    --wandb

# batchinvar1e-2 antiinvar0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder0129_multiprogram/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0127_multiprogram_batchinvar1e-2_antiinvar0.5 \
    --batch_invar_loss_lambda 1e-2 \
    --anti_invar_ratio 0.5 \
    --wandb

# batchinvar1e-1 antiinvar0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder0129_multiprogram/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0127_multiprogram_batchinvar1e-1_antiinvar0.5 \
    --batch_invar_loss_lambda 1e-1 \
    --anti_invar_ratio 0.5 \
    --wandb
