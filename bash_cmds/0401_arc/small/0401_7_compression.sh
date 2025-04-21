# python make_sbatch.py --ngpu 1 --time 1 --single --bash_files bash_cmds/0401_arc/small/0401_7_compression.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)


# arc compression0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_compression0.9 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --compression_ratio 0.9

# arc compression0.75
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_compression0.75 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --compression_ratio 0.75

# arc compression0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_compression0.5 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --compression_ratio 0.5

# arc compression0.25
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_compression0.25 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --compression_ratio 0.25

# Submitted batch job 59519232