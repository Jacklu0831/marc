# two runs of 2 hours and 3 hours

# arc gs200 lr3e-3 tokendrop0.1 part1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_gs200_lr3e-3_tokendrop0.1_part1_run2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 200 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --gs_float16

# arc gs200 lr3e-3 tokendrop0.1 part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_gs200_lr3e-3_tokendrop0.1_part2_run2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 200 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --gs_float16

# arc gs200 lr3e-3 tokendrop0.1 part3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos \
    --tag arc_gs200_lr3e-3_tokendrop0.1_part3_run2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 200 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --gs_float16

# arc gs200 lr3e-3 tokendrop0.1 part4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos \
    --tag arc_gs200_lr3e-3_tokendrop0.1_part4_run2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 200 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --gs_float16

# arc gs200 lr3e-3 tokendrop0.1 part5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos \
    --tag arc_gs200_lr3e-3_tokendrop0.1_part5_run2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 200 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --gs_float16
