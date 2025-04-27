# python make_sbatch.py --ngpu 1 --time 12 --single --bash_files bash_cmds/0401_arc/randomsearchfull/0.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# arc gs250 lr1e-2 randomkv token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr1e-2_randomkv_token \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 250 \
    --gs_lr 1e-2 \
    --random_kv token

# arc gs500 lr1e-2 randomkv token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs500_lr1e-2_randomkv_token \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 500 \
    --gs_lr 1e-2 \
    --random_kv token

# arc gs750 lr1e-2 randomkv token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs750_lr1e-2_randomkv_token \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 750 \
    --gs_lr 1e-2 \
    --random_kv token

# arc gs1000 lr1e-2 randomkv token
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs1000_lr1e-2_randomkv_token \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 1000 \
    --gs_lr 1e-2 \
    --random_kv token

# Submitted batch job 59824222