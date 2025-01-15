# python make_sbatch.py --ngpu 1 --time 4 --bash_files bash_commands/0112_encdec_eval/0112_3_eval_encoderloss1.0.sh

# 400task original
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --tag 400task_original \
    --weight_dir manual_copy_0111_encoderloss1.0_demonloss \
    --weight_epoch 13 \
    --data_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --compact_grids \
    --max_seq_len 5120

# 80task leavens0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --tag 80task_leavens0 \
    --weight_dir manual_copy_0111_encoderloss1.0_demonloss \
    --weight_epoch 13 \
    --data_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --select_tasks_path task_info_selected.csv \
    --compact_grids \
    --max_seq_len 5120 \
    --leave_ns 0 \
    --permute_n 2 \
    --augment_n 5 \
    --leave_ns_inc

# 80task leavens1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --tag 80task_leavens1 \
    --weight_dir manual_copy_0111_encoderloss1.0_demonloss \
    --weight_epoch 13 \
    --data_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --select_tasks_path task_info_selected.csv \
    --compact_grids \
    --max_seq_len 5120 \
    --leave_ns 1 \
    --permute_n 2 \
    --augment_n 5 \
    --leave_ns_inc

# 80task leavens01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder/evaluate.py \
    --tag 80task_leavens01 \
    --weight_dir manual_copy_0111_encoderloss1.0_demonloss \
    --weight_epoch 13 \
    --data_dir /scratch/yl11330/re-arc/arc_original/evaluation \
    --select_tasks_path task_info_selected.csv \
    --compact_grids \
    --max_seq_len 5120 \
    --leave_ns 0 1 \
    --permute_n 2 \
    --augment_n 5 \
    --leave_ns_inc

# competition accuracies:
# Submitted batch job 55808735 # 0.03
# Submitted batch job 55808023 # 0.0625
# Submitted batch job 55816199 # 0.05
# Submitted batch job 55816200 # 0.0625