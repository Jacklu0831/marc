# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0121/0121_6_invarloss_antiinvar_linear.sh
# instead of these, search in 0.001, 0.01, 0.1, etc

# invar0.03 antiinvarratio0.0 linear
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder7/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0121_invar0.03_antiinvarratio0.0_linear \
    --invar_loss_lambda 0.03 \
    --linear_invar \
    --anti_invar_ratio 0.0 \
    --wandb

# invar0.1 antiinvarratio0.0 linear
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder7/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0121_invar0.1_antiinvarratio0.0_linear \
    --invar_loss_lambda 0.1 \
    --linear_invar \
    --anti_invar_ratio 0.0 \
    --wandb

# invar0.03 antiinvarratio0.5 linear
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder7/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0121_invar0.03_antiinvarratio0.5_linear \
    --invar_loss_lambda 0.03 \
    --linear_invar \
    --anti_invar_ratio 0.5 \
    --wandb

# invar0.1 antiinvarratio0.5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder7/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0121_invar0.1_antiinvarratio0.5_linear \
    --invar_loss_lambda 0.1 \
    --linear_invar \
    --anti_invar_ratio 0.5 \
    --wandb
