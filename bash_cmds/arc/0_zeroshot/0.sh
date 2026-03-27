# run locally

# arc zeroshot part1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_zeroshot_part1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --zero_shot

# arc zeroshot part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_zeroshot_part2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --zero_shot

# arc zeroshot part3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos \
    --tag arc_zeroshot_part3 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --zero_shot

# arc zeroshot part4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos \
    --tag arc_zeroshot_part4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --zero_shot

# arc zeroshot part5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos \
    --tag arc_zeroshot_part5 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --zero_shot

# 0.0
# 0.0
# 0.0125
# 0.025
# 0.0125
# avg: 0.01