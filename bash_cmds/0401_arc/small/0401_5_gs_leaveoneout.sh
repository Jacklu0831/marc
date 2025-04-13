# python make_sbatch.py --ngpu 1 --time 4 --bash_files bash_cmds/0401_arc/small/0401_5_gs_leaveoneout.sh



# arc gs5 lr1e-2 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag test \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_leave_one_out

# arc gs25 lr1e-2 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr1e-2_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_leave_one_out

# arc gs100 lr1e-2 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-2_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_leave_one_out

# arc gs250 lr1e-2 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr1e-2_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_leave_one_out






# arc gs5 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr1e-3_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_leave_one_out

# arc gs25 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr1e-3_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_leave_one_out

# arc gs100 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-3_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_leave_one_out

# arc gs250 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr1e-3_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_leave_one_out







# arc gs5 lr1e-4 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr1e-4_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_leave_one_out

# arc gs25 lr1e-4 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr1e-4_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_leave_one_out

# arc gs100 lr1e-4 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-4_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_leave_one_out

# arc gs250 lr1e-4 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr1e-4_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_leave_one_out


# lr1e-2
# Submitted batch job 59237857
# Submitted batch job 59237858
# Submitted batch job 59237859
# Submitted batch job 59237860

# lr1e-3
# Submitted batch job 59237861
# Submitted batch job 59237862
# Submitted batch job 59237863
# Submitted batch job 59237864

# lr1e-4
# Submitted batch job 59237865
# Submitted batch job 59237866
# Submitted batch job 59237867
# Submitted batch job 59237868

# if results inconclusive, try full