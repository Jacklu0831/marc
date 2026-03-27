MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# arc gs50 lr5e-2 dropnone tokendrop0.1 ntoken32 part3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos \
    --tag arc_gs50_lr5e-2_dropnone_tokendrop0.1_ntoken32_part3 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 50 \
    --gs_lr 5e-2 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --gs_ntokens 32

# arc gs100 lr5e-2 dropnone tokendrop0.1 ntoken32 part3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos \
    --tag arc_gs100_lr5e-2_dropnone_tokendrop0.1_ntoken32_part3 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 100 \
    --gs_lr 5e-2 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --gs_ntokens 32

# arc gs150 lr5e-2 dropnone tokendrop0.1 ntoken32 part3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos \
    --tag arc_gs150_lr5e-2_dropnone_tokendrop0.1_ntoken32_part3 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 150 \
    --gs_lr 5e-2 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --gs_ntokens 32

# arc gs200 lr5e-2 dropnone tokendrop0.1 ntoken32 part3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos \
    --tag arc_gs200_lr5e-2_dropnone_tokendrop0.1_ntoken32_part3 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 200 \
    --gs_lr 5e-2 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --gs_ntokens 32

# arc gs250 lr5e-2 dropnone tokendrop0.1 ntoken32 part3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos \
    --tag arc_gs250_lr5e-2_dropnone_tokendrop0.1_ntoken32_part3 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 250 \
    --gs_lr 5e-2 \
    --gs_dropout none \
    --gs_token_dropout 0.1 \
    --gs_ntokens 32

# running