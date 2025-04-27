# python make_sbatch.py --ngpu 1 --time 14 --single --bash_files bash_cmds/0401_arc/tttsearch/0.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# arc ttt iter25
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_ttt_iter25 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 25 \
    --ttt_permute_n 2000

# arc ttt iter50
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_ttt_iter50 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 50 \
    --ttt_permute_n 2000

# arc ttt iter100
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_ttt_iter100 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 100 \
    --ttt_permute_n 2000

# arc ttt iter200
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_ttt_iter200 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 200 \
    --ttt_permute_n 2000

# arc ttt iter300
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_ttt_iter300 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 300 \
    --ttt_permute_n 2000

# arc ttt iter400
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_ttt_iter400 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 400 \
    --ttt_permute_n 2000

# arc ttt iter500
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_ttt_iter500 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_iters 500 \
    --ttt_permute_n 2000

# Submitted batch job 59824370