# python make_sbatch.py --ngpu 1 --time 2 --bash_files bash_cmds/0401_arc/small/0401_6_gs_droppowerwithtrain_detach.sh

# arc gs5 lr1e-3 droppowerwithtrain detach
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr1e-3_droppowerwithtrain_detach \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_dropout power_with_train \
    --gs_detach

# arc gs25 lr1e-3 droppowerwithtrain detach
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr1e-3_droppowerwithtrain_detach \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout power_with_train \
    --gs_detach

# arc gs100 lr1e-3 droppowerwithtrain detach
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-3_droppowerwithtrain_detach \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout power_with_train \
    --gs_detach

# arc gs250 lr1e-3 droppowerwithtrain detach
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr1e-3_droppowerwithtrain_detach \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --gs_dropout power_with_train \
    --gs_detach





# arc gs5 lr1e-4 droppowerwithtrain detach
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr1e-4_droppowerwithtrain_detach \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 5 \
    --gs_lr 1e-4 \
    --gs_dropout power_with_train \
    --gs_detach

# arc gs25 lr1e-4 droppowerwithtrain detach
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr1e-4_droppowerwithtrain_detach \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 25 \
    --gs_lr 1e-4 \
    --gs_dropout power_with_train \
    --gs_detach

# arc gs100 lr1e-4 droppowerwithtrain detach
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-4_droppowerwithtrain_detach \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 100 \
    --gs_lr 1e-4 \
    --gs_dropout power_with_train \
    --gs_detach

# arc gs250 lr1e-4 droppowerwithtrain detach
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr1e-4_droppowerwithtrain_detach \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 250 \
    --gs_lr 1e-4 \
    --gs_dropout power_with_train \
    --gs_detach

# Submitted batch job 59519336
# Submitted batch job 59519337
# Submitted batch job 59519338
# Submitted batch job 59519339
# Submitted batch job 59519340
# Submitted batch job 59519341
# Submitted batch job 59519342
# Submitted batch job 59519343