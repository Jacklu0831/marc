accelerate launch \
    --main_process_port $MASTER_PORT \
    --mixed_precision bf16 encoder_decoder1030_multiprogram/train.py \
    --tag test \
    --dry_train_run \
    --num_workers 0 \
    --train_batch_size 2 \
    --max_num_train_program 1

accelerate launch \
    --main_process_port $MASTER_PORT \
    --mixed_precision bf16 encoder_decoder1030_multiprogram/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag test \
    --dry_eval_run \
    --num_workers 0 \
    --train_batch_size 2 \
    --max_num_train_program 1 \
    --samples_per_epoch 8

accelerate launch \
    --main_process_port $MASTER_PORT \
    --mixed_precision bf16 encoder_decoder0124_debug1/train.py \
    --tag test \
    --dry_train_run \
    --num_workers 0 \
    --train_batch_size 2

accelerate launch \
    --main_process_port $MASTER_PORT \
    --mixed_precision bf16 encoder_decoder0124_debug1/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag test \
    --dry_eval_run \
    --num_workers 0 \
    --train_batch_size 2 \
    --samples_per_epoch 8
