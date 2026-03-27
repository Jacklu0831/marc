# NOTE: run 2 at a time on 4 hours interactive
# 4 runs with 400 tasks total to 260GB

# arc ttt iter150 save seed0 part1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_ttt_iter150_save_seed0_part1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 150 \
    --ttt_permute_n 1000 \
    --ttt_save \
    --seed 0

# arc ttt iter150 save seed1 part1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_ttt_iter150_save_seed1_part1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 150 \
    --ttt_permute_n 1000 \
    --ttt_save \
    --seed 1

# arc ttt iter150 save seed2 part1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_ttt_iter150_save_seed2_part1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 150 \
    --ttt_permute_n 1000 \
    --ttt_save \
    --seed 2

# arc ttt iter150 save seed3 part1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_ttt_iter150_save_seed3_part1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 150 \
    --ttt_permute_n 1000 \
    --ttt_save \
    --seed 3
