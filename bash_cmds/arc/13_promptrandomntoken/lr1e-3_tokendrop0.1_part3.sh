# arc prompt100 lr1e-3 dropnone tokendrop0.1 random part3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_prompt/test_time_evaluate.py \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos \
    --tag arc_prompt100_lr1e-3_dropnone_tokendrop0.1_random_part3 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32

# arc prompt150 lr1e-3 dropnone tokendrop0.1 random part3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_prompt/test_time_evaluate.py \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos \
    --tag arc_prompt150_lr1e-3_dropnone_tokendrop0.1_random_part3 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 150 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32

# arc prompt200 lr1e-3 dropnone tokendrop0.1 random part3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_prompt/test_time_evaluate.py \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos \
    --tag arc_prompt200_lr1e-3_dropnone_tokendrop0.1_random_part3 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 200 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32

# arc prompt250 lr1e-3 dropnone tokendrop0.1 random part3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_prompt/test_time_evaluate.py \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos \
    --tag arc_prompt250_lr1e-3_dropnone_tokendrop0.1_random_part3 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --random_prompt token \
    --random_prompt_ntokens 32
