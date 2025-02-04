# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0127_multiprogram/0127_0_multiprogram.sh

# base
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder0125_multiprogram/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0127_multiprogram_base \
    --lr_program 1e-5 \
    --wandb

# leftpad
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder0125_multiprogram/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0127_multiprogram_leftpad \
    --encoder_pad_side left \
    --decoder_pad_side left \
    --lr_program 1e-5 \
    --wandb

# programlr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder0125_multiprogram/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0127_multiprogram_programlr1e-4 \
    --lr_program 1e-4 \
    --wandb

# Submitted batch job 56566341
# Submitted batch job 56566342
# Submitted batch job 56566343