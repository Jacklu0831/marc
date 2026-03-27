# arc evalondemon part1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_evalondemon_part1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --eval_on_demonstrations

# arc evalondemon part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_evalondemon_part2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --eval_on_demonstrations

# arc evalondemon part3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos \
    --tag arc_evalondemon_part3 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --eval_on_demonstrations

# arc evalondemon part4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos \
    --tag arc_evalondemon_part4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --eval_on_demonstrations

# arc evalondemon part5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos \
    --tag arc_evalondemon_part5 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --eval_on_demonstrations

# this will take LONG and should really double check if correct (check the slurm outputs)
# normal 80 task takes 9min => 400 tasks and avg <4 demon per task => 180mins, 4hr should be fine

# 0.1
# 0.1125
# 0.05
# 0.075
# 0.0875

# why SO low? must be a bug

# exactacc
# 0.22519083969465647
# 0.2633587786259542
# 0.1588447653429603
# 0.2066420664206642
# 0.27636363636363637
# avg: 0.22608001728957