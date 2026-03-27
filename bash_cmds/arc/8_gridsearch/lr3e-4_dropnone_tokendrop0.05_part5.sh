# arc gs100 lr3e-4 dropnone tokendrop0.05 part5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos \
    --tag arc_gs100_lr3e-4_dropnone_tokendrop0.05_part5 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 100 \
    --gs_lr 3e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.05

# arc gs150 lr3e-4 dropnone tokendrop0.05 part5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos \
    --tag arc_gs150_lr3e-4_dropnone_tokendrop0.05_part5 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 150 \
    --gs_lr 3e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.05

# arc gs200 lr3e-4 dropnone tokendrop0.05 part5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos \
    --tag arc_gs200_lr3e-4_dropnone_tokendrop0.05_part5 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 200 \
    --gs_lr 3e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.05

# arc gs250 lr3e-4 dropnone tokendrop0.05 part5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos \
    --tag arc_gs250_lr3e-4_dropnone_tokendrop0.05_part5 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 250 \
    --gs_lr 3e-4 \
    --gs_dropout none \
    --gs_token_dropout 0.05
