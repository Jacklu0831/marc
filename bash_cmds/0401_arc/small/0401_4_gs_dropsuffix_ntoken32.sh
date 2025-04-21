# python make_sbatch.py --ngpu 1 --time 1 --bash_files bash_cmds/0401_arc/small/0401_4_gs_dropsuffix_ntoken32.sh

# arc gs5 lr1e-3 dropsuffix ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr1e-3_dropsuffix_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_dropout suffix \
    --gs_ntokens 32

# arc gs25 lr1e-3 dropsuffix ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr1e-3_dropsuffix_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout suffix \
    --gs_ntokens 32

# arc gs100 lr1e-3 dropsuffix ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-3_dropsuffix_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout suffix \
    --gs_ntokens 32

# arc gs250 lr1e-3 dropsuffix ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr1e-3_dropsuffix_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --gs_dropout suffix \
    --gs_ntokens 32





# arc gs5 lr1e-4 dropsuffix ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr1e-4_dropsuffix_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 5 \
    --gs_lr 1e-4 \
    --gs_dropout suffix \
    --gs_ntokens 32

# arc gs25 lr1e-4 dropsuffix ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr1e-4_dropsuffix_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 25 \
    --gs_lr 1e-4 \
    --gs_dropout suffix \
    --gs_ntokens 32

# arc gs100 lr1e-4 dropsuffix ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-4_dropsuffix_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 100 \
    --gs_lr 1e-4 \
    --gs_dropout suffix \
    --gs_ntokens 32

# arc gs250 lr1e-4 dropsuffix ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr1e-4_dropsuffix_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 250 \
    --gs_lr 1e-4 \
    --gs_dropout suffix \
    --gs_ntokens 32

# Submitted batch job 59519134
# Submitted batch job 59519135
# Submitted batch job 59519136
# Submitted batch job 59519137
# Submitted batch job 59519138
# Submitted batch job 59519139
# Submitted batch job 59519140
# Submitted batch job 59519141