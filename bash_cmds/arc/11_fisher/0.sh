# arc fisher part1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_fisher/test_time_evaluate.py \
    --select_tasks_path data/task_info_part1.csv \
    --no_bos \
    --tag arc_fisher_part1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 1 \
    --gs_fisher \
    --gs_lambda_param_sqr 1e-10 \
    --gs_log_fisher

# arc fisher part2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_fisher/test_time_evaluate.py \
    --select_tasks_path data/task_info_part2.csv \
    --no_bos \
    --tag arc_fisher_part2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 1 \
    --gs_fisher \
    --gs_lambda_param_sqr 1e-10 \
    --gs_log_fisher

# arc fisher part3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_fisher/test_time_evaluate.py \
    --select_tasks_path data/task_info_part3.csv \
    --no_bos \
    --tag arc_fisher_part3 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 1 \
    --gs_fisher \
    --gs_lambda_param_sqr 1e-10 \
    --gs_log_fisher

# arc fisher part4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_fisher/test_time_evaluate.py \
    --select_tasks_path data/task_info_part4.csv \
    --no_bos \
    --tag arc_fisher_part4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 1 \
    --gs_fisher \
    --gs_lambda_param_sqr 1e-10 \
    --gs_log_fisher

# arc fisher part5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_fisher/test_time_evaluate.py \
    --select_tasks_path data/task_info_part5.csv \
    --no_bos \
    --tag arc_fisher_part5 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 1 \
    --gs_fisher \
    --gs_lambda_param_sqr 1e-10 \
    --gs_log_fisher

# waiting to run this on a100, should be quick

# 'eval/gs_fisher_key': 1.457640919717463e-09,
# 'eval/gs_fisher_val': 5.2927625053645686e-08,
# 'eval/gs_fisher_key': 4.72565932972567e-09,
# 'eval/gs_fisher_val': 2.5453404330120417e-07,
# 'eval/gs_fisher_key': 1.1549346858012871e-09,
# 'eval/gs_fisher_val': 4.06839122287468e-08,
# 'eval/gs_fisher_key': 2.6193190634218782e-09,
# 'eval/gs_fisher_val': 9.812132547545313e-08,
# 'eval/gs_fisher_key': 2.188057802406992e-09,
# 'eval/gs_fisher_val': 6.526639267062423e-08,

# key: 2.4291223602147E-9
# val: 1.0230665974593E-7