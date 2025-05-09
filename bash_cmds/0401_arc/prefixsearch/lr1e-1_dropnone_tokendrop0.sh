# python make_sbatch.py --ngpu 1 --time 4 --single --bash_files bash_cmds/0401_arc/prefixsearch/lr1e-1_dropnone_tokendrop0.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# arc gs50 lr1e-1 dropnone tokendrop0 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs50_lr1e-1_dropnone_tokendrop0_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 50 \
    --gs_lr 1e-1 \
    --gs_dropout none \
    --gs_token_dropout 0.0 \
    --gs_ntokens 32

# arc gs100 lr1e-1 dropnone tokendrop0 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-1_dropnone_tokendrop0_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 100 \
    --gs_lr 1e-1 \
    --gs_dropout none \
    --gs_token_dropout 0.0 \
    --gs_ntokens 32

# arc gs150 lr1e-1 dropnone tokendrop0 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs150_lr1e-1_dropnone_tokendrop0_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 150 \
    --gs_lr 1e-1 \
    --gs_dropout none \
    --gs_token_dropout 0.0 \
    --gs_ntokens 32

# arc gs200 lr1e-1 dropnone tokendrop0 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs200_lr1e-1_dropnone_tokendrop0_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 200 \
    --gs_lr 1e-1 \
    --gs_dropout none \
    --gs_token_dropout 0.0 \
    --gs_ntokens 32

# arc gs250 lr1e-1 dropnone tokendrop0 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr1e-1_dropnone_tokendrop0_ntoken32 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 250 \
    --gs_lr 1e-1 \
    --gs_dropout none \
    --gs_token_dropout 0.0 \
    --gs_ntokens 32

# running