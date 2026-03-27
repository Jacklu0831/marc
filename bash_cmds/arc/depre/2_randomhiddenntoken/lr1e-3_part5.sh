MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# arc gs250 lr1e-3 randomhidden token part5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_hidden/test_time_evaluate.py \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos \
    --tag arc_gs250_lr1e-3_randomhidden_token_ntoken32_part5 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --random_hidden token \
    --random_hidden_ntokens 32

# arc gs500 lr1e-3 randomhidden token part5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_hidden/test_time_evaluate.py \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos \
    --tag arc_gs500_lr1e-3_randomhidden_token_ntoken32_part5 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 500 \
    --gs_lr 1e-3 \
    --random_hidden token \
    --random_hidden_ntokens 32

# arc gs750 lr1e-3 randomhidden token part5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_hidden/test_time_evaluate.py \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos \
    --tag arc_gs750_lr1e-3_randomhidden_token_ntoken32_part5 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 750 \
    --gs_lr 1e-3 \
    --random_hidden token \
    --random_hidden_ntokens 32

# arc gs1000 lr1e-3 randomhidden token part5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_hidden/test_time_evaluate.py \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos \
    --tag arc_gs1000_lr1e-3_randomhidden_token_ntoken32_part5 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 1000 \
    --gs_lr 1e-3 \
    --random_hidden token \
    --random_hidden_ntokens 32
