# ct-kv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_time_ctkv_iter10 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 10 \
    --gs_batch_size 100000

# ct-kv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_time_ctkv_iter50 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 50 \
    --gs_batch_size 100000

# prefix m32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_time_prefixm32_iter10 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 10 \
    --random_kv token \
    --random_kv_ntokens 32 \
    --gs_batch_size 100000

# prefix m32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_time_prefixm32_iter50 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 50 \
    --random_kv token \
    --random_kv_ntokens 32 \
    --gs_batch_size 100000

# prompt m32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_prompt_time/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_time_promptm32_iter10 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 10 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --gs_batch_size 100000

# prompt m32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_prompt_time/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_time_promptm32_iter50 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 50 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --gs_batch_size 100000

# 1.1336666578199805
# 5.747518926132016
# 0.7334891441391735
# 3.6766624247155537
# 0.757734551662352
# 3.787783590758719