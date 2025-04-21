# python make_sbatch.py --ngpu 1 --time 1 --bash_files bash_cmds/0401_arc/small/0401_3_gs_droptrain.sh

# arc gs5 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr1e-3_droptrain \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_dropout train

# arc gs25 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr1e-3_droptrain \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout train

# arc gs100 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-3_droptrain \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout train

# arc gs250 lr1e-3 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr1e-3_droptrain \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --gs_dropout train





# arc gs5 lr1e-4 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr1e-4_droptrain \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 5 \
    --gs_lr 1e-4 \
    --gs_dropout train

# arc gs25 lr1e-4 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr1e-4_droptrain \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 25 \
    --gs_lr 1e-4 \
    --gs_dropout train

# arc gs100 lr1e-4 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-4_droptrain \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 100 \
    --gs_lr 1e-4 \
    --gs_dropout train

# arc gs250 lr1e-4 droptrain
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr1e-4_droptrain \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 250 \
    --gs_lr 1e-4 \
    --gs_dropout train

# Submitted batch job 59519126
# Submitted batch job 59519127
# Submitted batch job 59519128
# Submitted batch job 59519129
# Submitted batch job 59519130
# Submitted batch job 59519131
# Submitted batch job 59519132
# Submitted batch job 59519133