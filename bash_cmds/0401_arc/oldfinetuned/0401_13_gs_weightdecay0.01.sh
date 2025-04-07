# python make_sbatch.py --ngpu 1 --time 2 --bash_files bash_cmds/0401_arc/oldfinetuned/0401_13_gs_weightdecay0.01.sh
# should give similar performance to the semi-successful experiments from before
# if not, compare model loss with original encoder_decoder_noprogram/evaluate.py
# for a 1:1 reproduction, gs_take_best and gs_max_grad_norm=1e8




# arc gs5 lr1e-3 wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --flash_attn \
    --tag arc_gs5_lr1e-3_wd0.01 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_weight_decay 0.01

# arc gs25 lr1e-3 wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --flash_attn \
    --tag arc_gs25_lr1e-3_wd0.01 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_weight_decay 0.01

# arc gs100 lr1e-3 wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --flash_attn \
    --tag arc_gs100_lr1e-3_wd0.01 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_weight_decay 0.01

# arc gs250 lr1e-3 wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --flash_attn \
    --tag arc_gs250_lr1e-3_wd0.01 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_weight_decay 0.01





# arc gs5 lr1e-4 wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --flash_attn \
    --tag arc_gs5_lr1e-4_wd0.01 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_weight_decay 0.01

# arc gs25 lr1e-4 wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --flash_attn \
    --tag arc_gs25_lr1e-4_wd0.01 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_weight_decay 0.01

# arc gs100 lr1e-4 wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --flash_attn \
    --tag arc_gs100_lr1e-4_wd0.01 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_weight_decay 0.01

# arc gs250 lr1e-4 wd0.01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --select_tasks_path task_info_selected.csv \
    --no_bos \
    --flash_attn \
    --tag arc_gs250_lr1e-4_wd0.01 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_weight_decay 0.01


# weightdecay0.01

# gslr1e-3
# Submitted batch job 59048028 # 0.225
# Submitted batch job 59048029 # 0.275
# Submitted batch job 59048030 # 0.2625
# Submitted batch job 59048031 # 0.2625

# gslr1e-4
# Submitted batch job 59048032 # 0.2
# Submitted batch job 59048033 # 0.2125
# Submitted batch job 59048034 # 0.2125
# Submitted batch job 59048035 # 0.275