# ctkv part1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_nparam_ctkv_iter1_part1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 1 \
    --gs_batch_size 100000

# ctkv part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_nparam_ctkv_iter1_part2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 1 \
    --gs_batch_size 100000

# ctkv part3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos \
    --tag arc_nparam_ctkv_iter1_part3 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 1 \
    --gs_batch_size 100000

# ctkv part4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos \
    --tag arc_nparam_ctkv_iter1_part4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 1 \
    --gs_batch_size 100000

# ctkv part5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos \
    --tag arc_nparam_ctkv_iter1_part5 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 1 \
    --gs_batch_size 100000





# prompt part1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_prompt_time/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_nparam_prompt1_part1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 1 \
    --gs_batch_size 4

# prompt part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_prompt_time/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_nparam_prompt1_part2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 1 \
    --gs_batch_size 4

# prompt part3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_prompt_time/test_time_evaluate.py \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos \
    --tag arc_nparam_prompt1_part3 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 1 \
    --gs_batch_size 4

# prompt part4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_prompt_time/test_time_evaluate.py \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos \
    --tag arc_nparam_prompt1_part4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 1 \
    --gs_batch_size 4

# prompt part5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_prompt_time/test_time_evaluate.py \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos \
    --tag arc_nparam_prompt1_part5 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 1 \
    --gs_batch_size 4

# ctkv
# 23670484.292682927
# 20804949.333333332
# 20902233.44578313
# 21453848.975609757
# 22887667.80952381
# avg 21943836.771387

# prompt
# 2958810.536585366
# 2600618.6666666665
# 2612779.1807228914
# 2681731.1219512196
# 2860958.476190476
# avg: 2742979.5964233