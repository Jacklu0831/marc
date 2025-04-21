# python make_sbatch.py --ngpu 1 --time 4 --bash_files bash_cmds/0401_arc/small/0401_7_gs_ntokens_leaveoneout.sh





# arc gs5 lr1e-1 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr1e-1_ntokens32_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-1 \
    --gs_ntokens 32 \
    --gs_leave_one_out

# arc gs25 lr1e-1 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr1e-1_ntokens32_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-1 \
    --gs_ntokens 32 \
    --gs_leave_one_out

# arc gs100 lr1e-1 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-1_ntokens32_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-1 \
    --gs_ntokens 32 \
    --gs_leave_one_out

# arc gs250 lr1e-1 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr1e-1_ntokens32_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-1 \
    --gs_ntokens 32 \
    --gs_leave_one_out







# arc gs5 lr1e-2 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr1e-2_ntokens32_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_ntokens 32 \
    --gs_leave_one_out

# arc gs25 lr1e-2 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr1e-2_ntokens32_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_ntokens 32 \
    --gs_leave_one_out

# arc gs100 lr1e-2 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-2_ntokens32_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_ntokens 32 \
    --gs_leave_one_out

# arc gs250 lr1e-2 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr1e-2_ntokens32_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_ntokens 32 \
    --gs_leave_one_out







# arc gs5 lr1e-3 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr1e-3_ntokens32_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_ntokens 32 \
    --gs_leave_one_out

# arc gs25 lr1e-3 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr1e-3_ntokens32_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_ntokens 32 \
    --gs_leave_one_out

# arc gs100 lr1e-3 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-3_ntokens32_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_ntokens 32 \
    --gs_leave_one_out

# arc gs250 lr1e-3 ntokens32 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr1e-3_ntokens32_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_ntokens 32 \
    --gs_leave_one_out


# lr1e-1
# Submitted batch job 59237932 # 0.2
# Submitted batch job 59237933 # 0.2375
# Submitted batch job 59237934 # 0.2125
# Submitted batch job 59237935 # 0.2

# lr1e-2
# Submitted batch job 59237936 # 0.1875
# Submitted batch job 59237937 # 0.1875
# Submitted batch job 59237938 # 0.2125
# Submitted batch job 59237939 # 0.2125

# lr1e-3
# Submitted batch job 59237940 # 0.1875
# Submitted batch job 59237941 # 0.1875
# Submitted batch job 59237942 # 0.1875
# Submitted batch job 59237943 # 0.1875

# so far 0.2375