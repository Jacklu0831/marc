# python make_sbatch.py --ngpu 1 --time 1 --bash_files bash_cmds/0401_arc/small/0419_2_gs_randomkv_lr1e-2.sh

# arc gs5 lr1e-2 randomkv normal ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr1e-2_randomkv_normal_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 5 \
    --gs_lr 1e-2 \
    --random_kv normal \
    --random_kv_ntokens 32

# arc gs25 lr1e-2 randomkv normal ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr1e-2_randomkv_normal_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 25 \
    --gs_lr 1e-2 \
    --random_kv normal \
    --random_kv_ntokens 32

# arc gs100 lr1e-2 randomkv normal ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-2_randomkv_normal_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 100 \
    --gs_lr 1e-2 \
    --random_kv normal \
    --random_kv_ntokens 32

# arc gs250 lr1e-2 randomkv normal ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr1e-2_randomkv_normal_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 250 \
    --gs_lr 1e-2 \
    --random_kv normal \
    --random_kv_ntokens 32











# arc gs5 lr1e-2 randomkv ntoken ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs5_lr1e-2_randomkv_ntoken_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 5 \
    --gs_lr 1e-2 \
    --random_kv token \
    --random_kv_ntokens 32

# arc gs25 lr1e-2 randomkv ntoken ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs25_lr1e-2_randomkv_ntoken_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 25 \
    --gs_lr 1e-2 \
    --random_kv token \
    --random_kv_ntokens 32

# arc gs100 lr1e-2 randomkv ntoken ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-2_randomkv_ntoken_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 100 \
    --gs_lr 1e-2 \
    --random_kv token \
    --random_kv_ntokens 32

# arc gs250 lr1e-2 randomkv ntoken ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr1e-2_randomkv_ntoken_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 250 \
    --gs_lr 1e-2 \
    --random_kv token \
    --random_kv_ntokens 32

# Submitted batch job 59519377
# Submitted batch job 59519378
# Submitted batch job 59519379
# Submitted batch job 59519380
# Submitted batch job 59519381
# Submitted batch job 59519382
# Submitted batch job 59519383
# Submitted batch job 59519384