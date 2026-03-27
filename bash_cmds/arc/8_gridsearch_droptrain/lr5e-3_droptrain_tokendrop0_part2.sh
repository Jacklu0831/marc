# arc gs50 lr5e-3 droptrain tokendrop0 part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_gs50_lr5e-3_droptrain_tokendrop0_part2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 50 \
    --gs_lr 5e-3 \
    --gs_dropout train \
    --gs_token_dropout 0

# arc gs100 lr5e-3 droptrain tokendrop0 part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_gs100_lr5e-3_droptrain_tokendrop0_part2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 100 \
    --gs_lr 5e-3 \
    --gs_dropout train \
    --gs_token_dropout 0

# arc gs150 lr5e-3 droptrain tokendrop0 part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_gs150_lr5e-3_droptrain_tokendrop0_part2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 150 \
    --gs_lr 5e-3 \
    --gs_dropout train \
    --gs_token_dropout 0

# arc gs200 lr5e-3 droptrain tokendrop0 part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_gs200_lr5e-3_droptrain_tokendrop0_part2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 200 \
    --gs_lr 5e-3 \
    --gs_dropout train \
    --gs_token_dropout 0

# arc gs250 lr5e-3 droptrain tokendrop0 part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_gs250_lr5e-3_droptrain_tokendrop0_part2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 250 \
    --gs_lr 5e-3 \
    --gs_dropout train \
    --gs_token_dropout 0
