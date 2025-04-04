# for eval 80 task with voting
# python make_sbatch.py --ngpu 1 --time 4 --bash_files bash_commands/0317_noprogram/0322_1_eval_ttt_voting.sh --rtx8000

# voting 0317_noprogram_base nottt leavens0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag nottt_leavens0 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --select_tasks_path task_info_selected.csv \
    --leave_ns 0 \
    --permute_n 2 \
    --augment_n 5 \
    --leave_ns_inc

# voting 0317_noprogram_base nottt leavens1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag nottt_leavens1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --select_tasks_path task_info_selected.csv \
    --leave_ns 1 \
    --permute_n 2 \
    --augment_n 5 \
    --leave_ns_inc





# voting 0317_noprogram_base augd8epoch10 leavens0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag augd8epoch10_leavens0 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augd8_0317_noprogram_base \
    --ttt_weight_epoch 10 \
    --select_tasks_path task_info_selected.csv \
    --leave_ns 0 \
    --permute_n 2 \
    --augment_n 5 \
    --leave_ns_inc

# voting 0317_noprogram_base augd8epoch10 leavens1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag augd8epoch10_leavens1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augd8_0317_noprogram_base \
    --ttt_weight_epoch 10 \
    --select_tasks_path task_info_selected.csv \
    --leave_ns 1 \
    --permute_n 2 \
    --augment_n 5 \
    --leave_ns_inc






# voting 0317_noprogram_base augextraepoch6 leavens0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag augextraepoch6_leavens0 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augextra_0317_noprogram_base \
    --ttt_weight_epoch 6 \
    --select_tasks_path task_info_selected.csv \
    --leave_ns 0 \
    --permute_n 2 \
    --augment_n 5 \
    --leave_ns_inc

# voting 0317_noprogram_base augextraepoch6 leavens1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 encoder_decoder_noprogram/evaluate.py \
    --no_bos \
    --tag augextraepoch6_leavens1 \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --ttt_weight_dir ttt_augextra_0317_noprogram_base \
    --ttt_weight_epoch 6 \
    --select_tasks_path task_info_selected.csv \
    --leave_ns 1 \
    --permute_n 2 \
    --augment_n 5 \
    --leave_ns_inc


# Submitted batch job 58849369 # 0.2375
# Submitted batch job 58849370 # 0.225
# Submitted batch job 58849371 # 0.375
# Submitted batch job 58881008 # 0.3375
# Submitted batch job 58849373 # 0.3625
# Submitted batch job 58849374 # 0.3625