# run all at a time

# arc ttt iter50 seed0 part3 nopermute
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos \
    --tag arc_ttt_iter50_seed0_part3_nopermute \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 50 \
    --ttt_permute_n 1 \
    --seed 0

# arc ttt iter100 seed0 part3 nopermute
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos \
    --tag arc_ttt_iter100_seed0_part3_nopermute \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 100 \
    --ttt_permute_n 1 \
    --seed 0

# arc ttt iter150 seed0 part3 nopermute
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos \
    --tag arc_ttt_iter150_seed0_part3_nopermute \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 150 \
    --ttt_permute_n 1 \
    --seed 0

# arc ttt iter200 seed0 part3 nopermute
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos \
    --tag arc_ttt_iter200_seed0_part3_nopermute \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 200 \
    --ttt_permute_n 1 \
    --seed 0