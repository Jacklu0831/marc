# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0119_multigpu/0113_4_fullshared_identity.sh
# 8gpus

# shared noidentity
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0119_shared_noidentity \
    --projection_type shared \
    --wandb

# full noidentity
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0119_full_noidentity \
    --projection_type full \
    --wandb

# shared identity
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0119_shared_identity \
    --projection_type shared \
    --identity_init \
    --wandb

# full identity
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0119_full_identity \
    --projection_type full \
    --identity_init \
    --wandb

# Submitted batch job 56073986
# Submitted batch job 56073987
# Submitted batch job 56073988
# Submitted batch job 56073989