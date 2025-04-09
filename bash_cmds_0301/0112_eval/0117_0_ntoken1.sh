# python make_sbatch.py --ngpu 1 --time 4 --bash_files bash_commands/0112_encdec_eval/0117_0_ntoken1.sh

# 400task original
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new4/evaluate.py \
    --tag 400task_original \
    --weight_dir manual_copy_0113_ntoken1 \
    --weight_epoch 14 \
    --data_dir ./data/re-arc/arc_original/evaluation \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --num_virtual_tokens 1

# 400task original noflashattn
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new4/evaluate.py \
    --tag 400task_original_noflashattn \
    --weight_dir manual_copy_0113_ntoken1 \
    --weight_epoch 14 \
    --data_dir ./data/re-arc/arc_original/evaluation \
    --compact_grids \
    --max_seq_len 5120 \
    --tie_models \
    --num_virtual_tokens 1

# 80task leavens0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new4/evaluate.py \
    --tag 80task_leavens0 \
    --weight_dir manual_copy_0113_ntoken1 \
    --weight_epoch 14 \
    --data_dir ./data/re-arc/arc_original/evaluation \
    --select_tasks_path task_info_selected.csv \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --num_virtual_tokens 1 \
    --leave_ns 0 \
    --permute_n 2 \
    --augment_n 5 \
    --leave_ns_inc

# 80task leavens1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new4/evaluate.py \
    --tag 80task_leavens1 \
    --weight_dir manual_copy_0113_ntoken1 \
    --weight_epoch 14 \
    --data_dir ./data/re-arc/arc_original/evaluation \
    --select_tasks_path task_info_selected.csv \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --num_virtual_tokens 1 \
    --leave_ns 1 \
    --permute_n 2 \
    --augment_n 5 \
    --leave_ns_inc

# 80task leavens01
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_new4/evaluate.py \
    --tag 80task_leavens01 \
    --weight_dir manual_copy_0113_ntoken1 \
    --weight_epoch 14 \
    --data_dir ./data/re-arc/arc_original/evaluation \
    --select_tasks_path task_info_selected.csv \
    --compact_grids \
    --max_seq_len 5120 \
    --flash_attn \
    --tie_models \
    --num_virtual_tokens 1 \
    --leave_ns 0 1 \
    --permute_n 2 \
    --augment_n 5 \
    --leave_ns_inc

# competition accuracies:
# Submitted batch job 55987406 # 0.02
# Submitted batch job 55987407 # 0.02
# Submitted batch job 55987408 # 0.05
# Submitted batch job 55987409 # 0.025
# Submitted batch job 55987410 # 0.025