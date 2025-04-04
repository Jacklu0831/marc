# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0121/0121_7_numpairsigma.sh

# sigma1 maxprefix9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder7/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0121_sigma1_maxprefix9 \
    --num_pair_sigma 1 \
    --max_prefix 9 \
    --wandb

# sigma1 maxprefix11
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder7/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0121_sigma1_maxprefix11 \
    --num_pair_sigma 1 \
    --max_prefix 11 \
    --wandb

# sigma2 maxprefix9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder7/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0121_sigma2_maxprefix9 \
    --num_pair_sigma 2 \
    --max_prefix 9 \
    --wandb

# sigma2 maxprefix11
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder7/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0121_sigma2_maxprefix11 \
    --num_pair_sigma 2 \
    --max_prefix 11 \
    --wandb
