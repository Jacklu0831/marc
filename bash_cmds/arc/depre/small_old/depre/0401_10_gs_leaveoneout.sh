# python make_sbatch.py --ngpu 1 --time 4 --bash_files bash_cmds/0401_arc/small/0401_10_gs_leaveoneout.sh





# arc gs5 lr1e-2 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr1e-2_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_leave_one_out

# arc gs25 lr1e-2 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr1e-2_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_leave_one_out

# arc gs100 lr1e-2 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-2_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_leave_one_out

# arc gs250 lr1e-2 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr1e-2_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_leave_one_out












# arc gs5 lr3e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr3e-3_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 3e-3 \
    --gs_leave_one_out

# arc gs25 lr3e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr3e-3_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 3e-3 \
    --gs_leave_one_out

# arc gs100 lr3e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr3e-3_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 3e-3 \
    --gs_leave_one_out

# arc gs250 lr3e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr3e-3_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 3e-3 \
    --gs_leave_one_out









# arc gs5 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr1e-3_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_leave_one_out

# arc gs25 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr1e-3_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_leave_one_out

# arc gs100 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-3_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_leave_one_out

# arc gs250 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr1e-3_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_leave_one_out

# arc gs325 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs300_lr1e-3_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 300 \
    --gs_lr 1e-3 \
    --gs_leave_one_out

# arc gs400 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs400_lr1e-3_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 400 \
    --gs_lr 1e-3 \
    --gs_leave_one_out





# arc gs5 lr1e-4 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr1e-4_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_leave_one_out

# arc gs25 lr1e-4 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr1e-4_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_leave_one_out

# arc gs100 lr1e-4 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-4_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_leave_one_out

# arc gs250 lr1e-4 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr1e-4_leaveoneout \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_leave_one_out



# lr1e-2
# Submitted batch job 59218938 # 0.225
# Submitted batch job 59218939 # 0.2
# Submitted batch job 59218940 # 0.1625
# Submitted batch job 59218941 # 0.15

# lr3e-3
# Submitted batch job 59218942 # 0.2125
# Submitted batch job 59218943 # 0.2375
# Submitted batch job 59218944 # 0.2125
# Submitted batch job 59218945 # 0.225

# lr1e-3
# Submitted batch job 59161739 # 0.1885
# Submitted batch job 59161740 # 0.2125
# Submitted batch job 59161741 # 0.2375
# Submitted batch job 59161742 # 0.2375
# Submitted batch job 59218953 # 0.2375
# Submitted batch job 59218954 # 0.2375

# lr1e-4
# Submitted batch job 59161743 # 0.1875
# Submitted batch job 59161744 # 0.2
# Submitted batch job 59161745 # 0.1875
# Submitted batch job 59161746 # 0.2

# so far 0.2375 -> maxposition!!!