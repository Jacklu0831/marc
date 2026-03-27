# arc prompt150 lr3e-3 dropnone tokendrop0.1 randomfull part5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_prompt/test_time_evaluate.py \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos \
    --tag arc_prompt150_lr3e-3_dropnone_tokendrop0.1_randomfull_part5 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 150 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --random_prompt token

# arc prompt100 lr3e-3 dropnone tokendrop0.1 randomfull part5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_prompt/test_time_evaluate.py \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos \
    --tag arc_prompt100_lr3e-3_dropnone_tokendrop0.1_randomfull_part5 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --random_prompt token

# arc prompt200 lr3e-3 dropnone tokendrop0.1 randomfull part5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_prompt/test_time_evaluate.py \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos \
    --tag arc_prompt200_lr3e-3_dropnone_tokendrop0.1_randomfull_part5 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 200 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --random_prompt token

# arc prompt250 lr3e-3 dropnone tokendrop0.1 randomfull part5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_prompt/test_time_evaluate.py \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos \
    --tag arc_prompt250_lr3e-3_dropnone_tokendrop0.1_randomfull_part5 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --random_prompt token
