# python make_sbatch.py --ngpu 1 --time 1 --bash_files bash_cmds/0401_arc/small/0401_6_gs_droppowerwithtrain_ntoken32.sh

# arc gs5 lr1e-3 droppowerwithtrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr1e-3_droppowerwithtrain_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_dropout power_with_train \
    --gs_ntokens 32

# arc gs25 lr1e-3 droppowerwithtrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr1e-3_droppowerwithtrain_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout power_with_train \
    --gs_ntokens 32

# arc gs100 lr1e-3 droppowerwithtrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-3_droppowerwithtrain_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout power_with_train \
    --gs_ntokens 32

# arc gs250 lr1e-3 droppowerwithtrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr1e-3_droppowerwithtrain_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --gs_dropout power_with_train \
    --gs_ntokens 32





# arc gs5 lr1e-4 droppowerwithtrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr1e-4_droppowerwithtrain_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 5 \
    --gs_lr 1e-4 \
    --gs_dropout power_with_train \
    --gs_ntokens 32

# arc gs25 lr1e-4 droppowerwithtrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr1e-4_droppowerwithtrain_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 25 \
    --gs_lr 1e-4 \
    --gs_dropout power_with_train \
    --gs_ntokens 32

# arc gs100 lr1e-4 droppowerwithtrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-4_droppowerwithtrain_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 100 \
    --gs_lr 1e-4 \
    --gs_dropout power_with_train \
    --gs_ntokens 32

# arc gs250 lr1e-4 droppowerwithtrain ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr1e-4_droppowerwithtrain_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 250 \
    --gs_lr 1e-4 \
    --gs_dropout power_with_train \
    --gs_ntokens 32

# Submitted batch job 59519344
# Submitted batch job 59519345
# Submitted batch job 59519346
# Submitted batch job 59519347
# Submitted batch job 59519348
# Submitted batch job 59519349
# Submitted batch job 59519350
# Submitted batch job 59519351