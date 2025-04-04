# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0119_multigpu/0113_5_prefix.sh
# 6gpus

# prefix shared ntokens8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0119_prefix2prefix_shared_ntokens8 \
    --conditioning_method prefix2prefix \
    --projection_type shared \
    --ntokens 8 \
    --wandb

# prefix full ntokens8
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0119_prefix2prefix_full_ntokens8 \
    --conditioning_method prefix2prefix \
    --projection_type full \
    --ntokens 8 \
    --wandb

# prefix shared ntokens16
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0119_prefix2prefix_shared_ntokens16 \
    --conditioning_method prefix2prefix \
    --projection_type shared \
    --ntokens 16 \
    --wandb

# Submitted batch job 56077345
# Submitted batch job 56077346
# Submitted batch job 56077347