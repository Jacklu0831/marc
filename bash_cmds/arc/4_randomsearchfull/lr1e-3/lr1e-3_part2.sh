MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# arc gs100 lr1e-3 randomkv token part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_gs100_lr1e-3_randomkv_token_part2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --random_kv token

# arc gs250 lr1e-3 randomkv token part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_gs250_lr1e-3_randomkv_token_part2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --random_kv token

# arc gs500 lr1e-3 randomkv token part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_gs500_lr1e-3_randomkv_token_part2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 500 \
    --gs_lr 1e-3 \
    --random_kv token

# arc gs750 lr1e-3 randomkv token part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_gs750_lr1e-3_randomkv_token_part2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 750 \
    --gs_lr 1e-3 \
    --random_kv token

# arc gs1000 lr1e-3 randomkv token part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_gs1000_lr1e-3_randomkv_token_part2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 1000 \
    --gs_lr 1e-3 \
    --random_kv token
