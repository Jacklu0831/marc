# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0119_multigpu/0113_1_invarloss.sh
# 16gpus

# invar0.03 antiinvarratio0.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0119_invar0.03_antiinvarratio0.0 \
    --invar_loss_lambda 0.03 \
    --wandb

# invar0.1 antiinvarratio0.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0119_invar0.1_antiinvarratio0.0 \
    --invar_loss_lambda 0.1 \
    --wandb

# invar0.3 antiinvarratio0.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0119_invar0.3_antiinvarratio0.0 \
    --invar_loss_lambda 0.3 \
    --wandb

# invar0.5 antiinvarratio0.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0119_invar0.5_antiinvarratio0.0 \
    --invar_loss_lambda 0.5 \
    --wandb

# didnt work, only invar0.03 showed any promise and still not as good as base
# Submitted batch job 56073969
# Submitted batch job 56073970
# Submitted batch job 56073971
# Submitted batch job 56073972
