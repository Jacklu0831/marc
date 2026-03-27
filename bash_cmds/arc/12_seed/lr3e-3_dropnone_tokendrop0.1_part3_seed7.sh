# arc gs50 lr3e-3 dropnone tokendrop0.1 part3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_permute/test_time_evaluate.py \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos \
    --tag arc_gs50_lr3e-3_dropnone_tokendrop0.1_part3_seed7 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 50 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --seed 7

# arc gs100 lr3e-3 dropnone tokendrop0.1 part3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_permute/test_time_evaluate.py \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos \
    --tag arc_gs100_lr3e-3_dropnone_tokendrop0.1_part3_seed7 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --seed 7

# arc gs150 lr3e-3 dropnone tokendrop0.1 part3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_permute/test_time_evaluate.py \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos \
    --tag arc_gs150_lr3e-3_dropnone_tokendrop0.1_part3_seed7 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 150 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --seed 7

# arc gs200 lr3e-3 dropnone tokendrop0.1 part3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_permute/test_time_evaluate.py \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos \
    --tag arc_gs200_lr3e-3_dropnone_tokendrop0.1_part3_seed7 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 200 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --seed 7

# arc gs250 lr3e-3 dropnone tokendrop0.1 part3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_permute/test_time_evaluate.py \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos \
    --tag arc_gs250_lr3e-3_dropnone_tokendrop0.1_part3_seed7 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --seed 7
