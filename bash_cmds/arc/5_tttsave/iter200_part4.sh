# NOTE: run 2 at a time on 4 hours interactive
# 4 runs with 400 tasks total to 260GB

# arc ttt iter200 save seed0 part4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos \
    --tag arc_ttt_iter200_save_seed0_part4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 200 \
    --ttt_permute_n 1000 \
    --ttt_save \
    --seed 0

# arc ttt iter200 save seed1 part4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos \
    --tag arc_ttt_iter200_save_seed1_part4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 200 \
    --ttt_permute_n 1000 \
    --ttt_save \
    --seed 1

# arc ttt iter200 save seed2 part4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos \
    --tag arc_ttt_iter200_save_seed2_part4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 200 \
    --ttt_permute_n 1000 \
    --ttt_save \
    --seed 2

# arc ttt iter200 save seed3 part4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos \
    --tag arc_ttt_iter200_save_seed3_part4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 200 \
    --ttt_permute_n 1000 \
    --ttt_save \
    --seed 3
