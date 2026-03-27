# python make_sbatch.py --ngpu 1 --time 2 --bash_files bash_cmds/0401_arc/oldfinetuned/0401_10_gs_gradnorm1e8.sh


# arc gs5 lr1e-3 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --flash_attn \
    --tag arc_gs5_lr1e-3_gradnorm1e8 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_max_grad_norm 1e8

# arc gs25 lr1e-3 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --flash_attn \
    --tag arc_gs25_lr1e-3_gradnorm1e8 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_max_grad_norm 1e8

# arc gs100 lr1e-3 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --flash_attn \
    --tag arc_gs100_lr1e-3_gradnorm1e8 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_max_grad_norm 1e8

# arc gs250 lr1e-3 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --flash_attn \
    --tag arc_gs250_lr1e-3_gradnorm1e8 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_max_grad_norm 1e8





# arc gs5 lr1e-4 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --flash_attn \
    --tag arc_gs5_lr1e-4_gradnorm1e8 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_max_grad_norm 1e8

# arc gs25 lr1e-4 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --flash_attn \
    --tag arc_gs25_lr1e-4_gradnorm1e8 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_max_grad_norm 1e8

# arc gs100 lr1e-4 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --flash_attn \
    --tag arc_gs100_lr1e-4_gradnorm1e8 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_max_grad_norm 1e8

# arc gs250 lr1e-4 gradnorm1e8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --flash_attn \
    --tag arc_gs250_lr1e-4_gradnorm1e8 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_max_grad_norm 1e8


# lr1e-3
# Submitted batch job 59078885 # 0.225
# Submitted batch job 59078886 # 0.275
# Submitted batch job 59078887 # 0.2625
# Submitted batch job 59078888 # 0.2625

# lr1e-4
# Submitted batch job 59078889 # 0.1875
# Submitted batch job 59078890 # 0.2125
# Submitted batch job 59078891 # 0.2125
# Submitted batch job 59078892 # 0.25