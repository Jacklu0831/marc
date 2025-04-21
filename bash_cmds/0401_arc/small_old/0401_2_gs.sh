# python make_sbatch.py --ngpu 1 --time 2 --bash_files bash_cmds/0401_arc/small/0401_2_gs.sh
# should give similar performance to the semi-successful experiments from before
# if not, compare model loss with original encoder_decoder_noprogram/evaluate.py
# for a 1:1 reproduction, gs_take_best and gs_max_grad_norm=1e8

# for >250 step
# python make_sbatch.py --ngpu 1 --time 4 --bash_files bash_cmds/0401_arc/small/0401_2_gs.sh




# arc gs5 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr1e-3 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-3

# arc gs25 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr1e-3 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-3

# arc gs100 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-3 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-3

# arc gs250 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr1e-3 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-3





# arc gs5 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr1e-4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-4

# arc gs25 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr1e-4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-4

# arc gs100 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-4

# arc gs250 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr1e-4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-4

# arc gs300 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs300_lr1e-4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 300 \
    --gs_lr 1e-4

# arc gs350 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs350_lr1e-4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 350 \
    --gs_lr 1e-4


# lr1e-3
# Submitted batch job 59139652 # 0.225
# Submitted batch job 59139653 # 0.2625
# Submitted batch job 59139654 # 0.25
# Submitted batch job 59139655 # 0.25

# lr1e-4
# Submitted batch job 59139656 # 0.1875
# Submitted batch job 59139657 # 0.2125
# Submitted batch job 59139658 # 0.2625
# Submitted batch job 59139659 # 0.275
# Submitted batch job 59219340 # 0.275
# Submitted batch job 59219341 # 0.275

# so far 0.275
