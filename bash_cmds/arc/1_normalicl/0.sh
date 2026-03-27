# run locally

# arc part1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_part1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24

# arc part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_part2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24

# arc part3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos \
    --tag arc_part3 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24

# arc part4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos \
    --tag arc_part4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24

# arc part5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos \
    --tag arc_part5 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24

# TO REPLACE
# 0.0875
# 0.1875
# 0.0875
# 0.0875
# 0.2125
# => average 0.1325