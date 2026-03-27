# NOTE: run 2 at a time on 4 hours interactive
# 4 runs with 400 tasks total to 260GB

# arc ttt iter200 save seed0 part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_ttt_iter200_save_seed0_part2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 200 \
    --ttt_permute_n 1000 \
    --ttt_save \
    --seed 0

# arc ttt iter200 save seed1 part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_ttt_iter200_save_seed1_part2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 200 \
    --ttt_permute_n 1000 \
    --ttt_save \
    --seed 1

# arc ttt iter200 save seed2 part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_ttt_iter200_save_seed2_part2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 200 \
    --ttt_permute_n 1000 \
    --ttt_save \
    --seed 2

# arc ttt iter200 save seed3 part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_ttt_iter200_save_seed3_part2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 200 \
    --ttt_permute_n 1000 \
    --ttt_save \
    --seed 3
