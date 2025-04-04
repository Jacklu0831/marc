# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0127_multiprogram/0127_3_multiprogram_fewepoch.sh

# epoch15
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder0125_multiprogram/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0127_multiprogram_numepoch15 \
    --num_epochs 15 \
    --wandb

# epoch20
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder0125_multiprogram/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0127_multiprogram_numepoch20 \
    --num_epochs 20 \
    --wandb

# Submitted batch job 56611833
# Submitted batch job 56611834