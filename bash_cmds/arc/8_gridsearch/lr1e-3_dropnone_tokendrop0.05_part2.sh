# arc gs100 lr1e-3 dropnone tokendrop0.05 part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_gs100_lr1e-3_dropnone_tokendrop0.05_part2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.05

# arc gs150 lr1e-3 dropnone tokendrop0.05 part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_gs150_lr1e-3_dropnone_tokendrop0.05_part2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 150 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.05

# arc gs200 lr1e-3 dropnone tokendrop0.05 part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_gs200_lr1e-3_dropnone_tokendrop0.05_part2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 200 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.05

# arc gs250 lr1e-3 dropnone tokendrop0.05 part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_gs250_lr1e-3_dropnone_tokendrop0.05_part2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.05

# 0.275
# 0.2625
# 0.2625
# 0.275