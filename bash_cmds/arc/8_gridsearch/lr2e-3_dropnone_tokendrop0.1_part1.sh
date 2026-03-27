# arc gs100 lr2e-3 dropnone tokendrop0.1 part1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_gs100_lr2e-3_dropnone_tokendrop0.1_part1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 100 \
    --gs_lr 2e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1

# arc gs150 lr2e-3 dropnone tokendrop0.1 part1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_gs150_lr2e-3_dropnone_tokendrop0.1_part1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 150 \
    --gs_lr 2e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1

# arc gs200 lr2e-3 dropnone tokendrop0.1 part1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_gs200_lr2e-3_dropnone_tokendrop0.1_part1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 200 \
    --gs_lr 2e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1

# arc gs250 lr2e-3 dropnone tokendrop0.1 part1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_gs250_lr2e-3_dropnone_tokendrop0.1_part1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 250 \
    --gs_lr 2e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1

# 0.1625
# 0.175
# 0.175
# 0.1625