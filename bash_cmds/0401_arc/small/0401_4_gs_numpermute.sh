# python make_sbatch.py --ngpu 1 --time 2 --bash_files bash_cmds/0401_arc/small/0401_15_gs_numpermute.sh





# arc gs5 lr3e-2 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr3e-2_permuten1024 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 3e-2 \
    --gs_num_permute 1024

# arc gs25 lr3e-2 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr3e-2_permuten1024 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 3e-2 \
    --gs_num_permute 1024

# arc gs100 lr3e-2 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr3e-2_permuten1024 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 3e-2 \
    --gs_num_permute 1024

# arc gs250 lr3e-2 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr3e-2_permuten1024 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 3e-2 \
    --gs_num_permute 1024








# arc gs5 lr1e-2 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr1e-2_permuten1024 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024

# arc gs25 lr1e-2 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr1e-2_permuten1024 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024

# arc gs100 lr1e-2 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-2_permuten1024 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024

# arc gs250 lr1e-2 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr1e-2_permuten1024 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024









# arc gs5 lr3e-3 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr3e-3_permuten1024 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 3e-3 \
    --gs_num_permute 1024

# arc gs25 lr3e-3 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr3e-3_permuten1024 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 3e-3 \
    --gs_num_permute 1024

# arc gs100 lr3e-3 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr3e-3_permuten1024 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 3e-3 \
    --gs_num_permute 1024

# arc gs250 lr3e-3 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr3e-3_permuten1024 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 3e-3 \
    --gs_num_permute 1024









# arc gs5 lr1e-3 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr1e-3_permuten1024 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024

# arc gs25 lr1e-3 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr1e-3_permuten1024 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024

# arc gs100 lr1e-3 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-3_permuten1024 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024

# arc gs250 lr1e-3 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr1e-3_permuten1024 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024


# lr3e-2
# Submitted batch job 59173825 # 0.15
# Submitted batch job 59173826 # 0.1
# Submitted batch job 59173827 # 0.125
# Submitted batch job 59173828 # 0.1375

# lr1e-2
# Submitted batch job 59173829 # 0.15
# Submitted batch job 59173830 # 0.2
# Submitted batch job 59173831 # 0.2
# Submitted batch job 59173832 # 0.1875

# lr3e-3
# Submitted batch job 59173833 # 0.1375
# Submitted batch job 59173834 # 0.1875
# Submitted batch job 59173835 # 0.25
# Submitted batch job 59173836 # 0.25

# lr1e-3
# Submitted batch job 59173837 # 0.125
# Submitted batch job 59173838 # 0.1625
# Submitted batch job 59173839 # 0.2
# Submitted batch job 59173840 # 0.2125

# sadly, nah doesnt improve, at most equal to normal gs