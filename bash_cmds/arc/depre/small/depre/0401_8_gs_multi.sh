# python make_sbatch.py --ngpu 1 --time 6 --bash_files bash_cmds/0401_arc/small/0401_8_gs_multi.sh
# each gs is 14min, multigs 16 -> 4hr

# arc gs100 lr1e-3 multi1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_multigs/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-3_multi1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 3e-3 \
    --gs_multi 1

# arc gs100 lr1e-3 multi4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_multigs/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-3_multi4 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 3e-3 \
    --gs_multi 4

# arc gs100 lr1e-3 multi16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc_multigs/test_time_evaluate.py \
    --select_tasks_path data/task_info_selected.csv \
    --no_bos \
    --tag arc_gs100_lr1e-3_multi16 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 3e-3 \
    --gs_multi 16

# Submitted batch job 59252021 # 0.25
# Submitted batch job 59245299 # 0.175
# Submitted batch job 59245300 # 0.15