# ttt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_time_ttt_iter10 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 10 \
    --ttt_permute_n 1000 \
    --ttt_batch_size 4 \
    --seed 0

# ttt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_time_ttt_iter50 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 50 \
    --ttt_permute_n 1000 \
    --ttt_batch_size 4 \
    --seed 0

# prompt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_prompt_time/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_time_prompt10 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 10 \
    --gs_dropout none \
    --gs_batch_size 4

# prompt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_prompt_time/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_time_prompt50 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 50 \
    --gs_dropout none \
    --gs_batch_size 4

# 3.159265896169151
# 12.582826271289733
# 2.7902126108727803
# 14.328669818436227