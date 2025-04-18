# python make_sbatch.py --ngpu 1 --time 10 --bash_files bash_cmds/0401_arc/full/0401_7_gs_ntokens_leaveoneout.sh





# arc gs5 lr1e-1 ntokens32 leaveoneout full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs5_lr1e-1_ntokens32_leaveoneout_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-1 \
    --gs_ntokens 32 \
    --gs_leave_one_out

# arc gs25 lr1e-1 ntokens32 leaveoneout full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs25_lr1e-1_ntokens32_leaveoneout_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-1 \
    --gs_ntokens 32 \
    --gs_leave_one_out

# arc gs100 lr1e-1 ntokens32 leaveoneout full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs100_lr1e-1_ntokens32_leaveoneout_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-1 \
    --gs_ntokens 32 \
    --gs_leave_one_out

# arc gs250 lr1e-1 ntokens32 leaveoneout full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs250_lr1e-1_ntokens32_leaveoneout_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-1 \
    --gs_ntokens 32 \
    --gs_leave_one_out







# arc gs5 lr1e-2 ntokens32 leaveoneout full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs5_lr1e-2_ntokens32_leaveoneout_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_ntokens 32 \
    --gs_leave_one_out

# arc gs25 lr1e-2 ntokens32 leaveoneout full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs25_lr1e-2_ntokens32_leaveoneout_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_ntokens 32 \
    --gs_leave_one_out

# arc gs100 lr1e-2 ntokens32 leaveoneout full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs100_lr1e-2_ntokens32_leaveoneout_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_ntokens 32 \
    --gs_leave_one_out

# arc gs250 lr1e-2 ntokens32 leaveoneout full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs250_lr1e-2_ntokens32_leaveoneout_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_ntokens 32 \
    --gs_leave_one_out







# arc gs5 lr1e-3 ntokens32 leaveoneout full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs5_lr1e-3_ntokens32_leaveoneout_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_ntokens 32 \
    --gs_leave_one_out

# arc gs25 lr1e-3 ntokens32 leaveoneout full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs25_lr1e-3_ntokens32_leaveoneout_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_ntokens 32 \
    --gs_leave_one_out

# arc gs100 lr1e-3 ntokens32 leaveoneout full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs100_lr1e-3_ntokens32_leaveoneout_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_ntokens 32 \
    --gs_leave_one_out

# arc gs250 lr1e-3 ntokens32 leaveoneout full
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_arc/test_time_evaluate.py \
    --no_bos \
    --tag arc_gs250_lr1e-3_ntokens32_leaveoneout_full \
    --weight_dir 0317_noprogram_base \
    --weight_epoch 24 \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_ntokens 32 \
    --gs_leave_one_out


# Submitted batch job 59293675 # 0.14
# Submitted batch job 59293676 # 0.175
# Submitted batch job 59293677 # 0.17
# Submitted batch job 59293678 # 0.1675

# Submitted batch job 59293679 # 0.1325
# Submitted batch job 59293680 # 0.1375
# Submitted batch job 59293681 # 0.1575
# Submitted batch job 59293682 # 0.185

# Submitted batch job 59293683 # 0.1325
# Submitted batch job 59293684 # 0.135
# Submitted batch job 59293685 # 0.135
# Submitted batch job 59293686 # 0.135

# so far 0.185