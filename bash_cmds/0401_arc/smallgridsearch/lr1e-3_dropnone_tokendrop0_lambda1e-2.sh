# python make_sbatch.py --ngpu 1 --time 4 --single --bash_files bash_cmds/0401_arc/smallgridsearch/lr1e-3_dropnone_tokendrop0_lambda1e-2.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# arc gs10 lr1e-3 dropnone tokendrop0 lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs10_lr1e-3_dropnone_tokendrop0_lambda1e-2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.0 \
    --gs_lambda_param_sqr 1e-2

# arc gs50 lr1e-3 dropnone tokendrop0 lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs50_lr1e-3_dropnone_tokendrop0_lambda1e-2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.0 \
    --gs_lambda_param_sqr 1e-2

# arc gs100 lr1e-3 dropnone tokendrop0 lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-3_dropnone_tokendrop0_lambda1e-2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.0 \
    --gs_lambda_param_sqr 1e-2

# arc gs150 lr1e-3 dropnone tokendrop0 lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs150_lr1e-3_dropnone_tokendrop0_lambda1e-2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 150 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.0 \
    --gs_lambda_param_sqr 1e-2

# arc gs200 lr1e-3 dropnone tokendrop0 lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs200_lr1e-3_dropnone_tokendrop0_lambda1e-2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 200 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.0 \
    --gs_lambda_param_sqr 1e-2

# arc gs250 lr1e-3 dropnone tokendrop0 lambda1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --tag arc_gs250_lr1e-3_dropnone_tokendrop0_lambda1e-2 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_epochs 250 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --gs_token_dropout 0.0 \
    --gs_lambda_param_sqr 1e-2

# 0.2375
# 0.25
# 0.25
# 0.25
# 0.25
# 0.25