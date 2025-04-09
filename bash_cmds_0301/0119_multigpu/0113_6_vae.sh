# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0119_multigpu/0113_6_vae.sh
# 8gpus

# vae shared noidentity
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0119_vae_shared_noidentity \
    --vae \
    --projection_type shared \
    --wandb

# vae full noidentity
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0119_vae_full_noidentity \
    --vae \
    --projection_type full \
    --wandb

# vae shared identity
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0119_vae_shared_identity \
    --vae \
    --projection_type shared \
    --identity_init \
    --wandb

# vae full identity
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0119_vae_full_identity \
    --vae \
    --projection_type full \
    --identity_init \
    --wandb

# scancelled, kl increases??
# Submitted batch job 56077348
# Submitted batch job 56077349
# Submitted batch job 56077350
# Submitted batch job 56077351