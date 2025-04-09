# python make_sbatch.py --ngpu 2 --time 48 --bash_files bash_commands/0121_singleprogram/0121_0_newformat.sh

# base
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder7/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0121_base \
    --wandb

# encleftpad
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder7/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0121_encleftpad \
    --encoder_pad_side left \
    --wandb

# encleftpad_decleftpad
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder7/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0121_encleftpad_decleftpad \
    --encoder_pad_side left \
    --decoder_pad_side left \
    --wandb

# decleftpad
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder7/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0121_decleftpad \
    --decoder_pad_side left \
    --wandb

# quantized4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder7/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0121_quantized4 \
    --untrainable_nbit 4 \
    --wandb

# uniformlr
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder7/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0121_uniformlr \
    --lr_embedding 1e-4 \
    --wandb

# weight decay
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder7/train.py \
    --eval_eval_leave_ns 0 \
    --eval_eval_permute_n 2 \
    --eval_eval_augment_n 5 \
    --eval_eval_leave_ns_inc \
    --eval_eval_select_tasks_path task_info_selected.csv \
    --tag 0121_weightdecay \
    --weight_decay 1e-2 \
    --wandb
