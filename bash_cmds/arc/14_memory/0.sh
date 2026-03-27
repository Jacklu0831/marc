MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# prompt32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_prompt/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_prompt32_memory \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 1 \
    --gs_batch_size 1 \
    --random_prompt token \
    --random_prompt_ntokens 32

# promptdemo
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_prompt/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_promptdemo_memory \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 1 \
    --gs_batch_size 1 \
    --random_prompt token

# prefix32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_prefix32_memory \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 1 \
    --gs_batch_size 1 \
    --random_kv token \
    --random_kv_ntokens 32

# prefixdemo
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_prefixdemo_memory \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 1 \
    --gs_batch_size 1 \
    --random_kv token

# ttt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_ttt_memory \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 1 \
    --ttt_batch_size 1 \
    --ttt_permute_n 1000

# ctprompt
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_prompt/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_ctprompt_memory \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 1 \
    --gs_batch_size 1 \
    --gs_token_dropout 0.1

# ctkv
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_ctkv_memory \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 1 \
    --gs_batch_size 1 \
    --gs_token_dropout 0.1

# 4486.123219559832
# 6451.220131478659
# 4598.243420112423
# 5340.345345846036
# 7232.5927138910065
# 6629.13876714939
# 5340.109059403582
