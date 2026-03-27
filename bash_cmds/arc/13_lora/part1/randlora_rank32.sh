MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# randlora rank32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag test \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --zero_shot \
    --gs_epochs 50 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_randlora \
    --gs_lora_rank 32 \
    --gs_lora_alpha 640 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 42

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag test \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --zero_shot \
    --gs_epochs 100 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_randlora \
    --gs_lora_rank 32 \
    --gs_lora_alpha 640 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 42

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag test \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --zero_shot \
    --gs_epochs 150 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_randlora \
    --gs_lora_rank 32 \
    --gs_lora_alpha 640 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 42

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag test \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --zero_shot \
    --gs_epochs 200 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_randlora \
    --gs_lora_rank 32 \
    --gs_lora_alpha 640 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 42

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag test \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --zero_shot \
    --gs_epochs 250 \
    --gs_dropout none \
    --gs_lora \
    --gs_lora_randlora \
    --gs_lora_rank 32 \
    --gs_lora_alpha 640 \
    --random_kv uniform \
    --random_kv_ntokens 0 \
    --seed 42
