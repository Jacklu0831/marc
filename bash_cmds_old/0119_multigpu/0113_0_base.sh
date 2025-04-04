# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0119_multigpu/0113_0_base.sh
# 12gpus

# base
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0119_base \
    --wandb

# tiemodels
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0119_tiemodels \
    --tie_models \
    --wandb

# weightdecay
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0119_weightdecay \
    --weight_decay 0.01 \
    --wandb

# quantized3.6
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0119_quantized3.6 \
    --untrainable_nbit 3.6 \
    --wandb

# quantized4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0119_quantized4 \
    --untrainable_nbit 4 \
    --wandb

# OOM
# quantized8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0119_quantized8 \
    --untrainable_nbit 8 \
    --wandb

# Submitted batch job 56073964 # base
# Submitted batch job 56073965 # tiemodels
# Submitted batch job 56142522 # CANCELLED before running
# Submitted batch job 56142523 # CANCELLED before running
# Submitted batch job 56073967 # quantized4