MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# arc prompt100 lr3e-3 dropnone tokendrop0.1 random part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_prompt_time/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_prompt100_lr3e-3_dropnone_tokendrop0.1_random_part2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 100 \
    --gs_lr 3e-3 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --gs_batch_size 100000

# arc prompt150 lr3e-3 dropnone tokendrop0.1 random part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_prompt_time/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_prompt150_lr3e-3_dropnone_tokendrop0.1_random_part2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 150 \
    --gs_lr 3e-3 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --gs_batch_size 100000

# arc prompt200 lr3e-3 dropnone tokendrop0.1 random part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_prompt_time/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_prompt200_lr3e-3_dropnone_tokendrop0.1_random_part2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 200 \
    --gs_lr 3e-3 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --gs_batch_size 100000

# arc prompt250 lr3e-3 dropnone tokendrop0.1 random part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_prompt_time/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_prompt250_lr3e-3_dropnone_tokendrop0.1_random_part2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 250 \
    --gs_lr 3e-3 \
    --random_prompt token \
    --random_prompt_ntokens 32 \
    --gs_batch_size 100000
