# python make_sbatch.py --ngpu 1 --time 1 --bash_files bash_cmds/0401_bbh/llama1b/0401_10_gs_leaveoneout.sh

# bbh llama1b gs5 lr1e-2 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs5_lr1e-2_leaveoneout \
    --model_name llama1b \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_leave_one_out

# bbh llama1b gs25 lr1e-2 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs25_lr1e-2_leaveoneout \
    --model_name llama1b \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_leave_one_out

# bbh llama1b gs100 lr1e-2 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs100_lr1e-2_leaveoneout \
    --model_name llama1b \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_leave_one_out

# bbh llama1b gs250 lr1e-2 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs250_lr1e-2_leaveoneout \
    --model_name llama1b \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_leave_one_out






# bbh llama1b gs5 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs5_lr1e-3_leaveoneout \
    --model_name llama1b \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_leave_one_out

# bbh llama1b gs25 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs25_lr1e-3_leaveoneout \
    --model_name llama1b \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_leave_one_out

# bbh llama1b gs100 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs100_lr1e-3_leaveoneout \
    --model_name llama1b \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_leave_one_out

# bbh llama1b gs250 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs250_lr1e-3_leaveoneout \
    --model_name llama1b \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_leave_one_out







# bbh llama1b gs5 lr1e-4 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs5_lr1e-4_leaveoneout \
    --model_name llama1b \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_leave_one_out

# bbh llama1b gs25 lr1e-4 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs25_lr1e-4_leaveoneout \
    --model_name llama1b \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_leave_one_out

# bbh llama1b gs100 lr1e-4 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs100_lr1e-4_leaveoneout \
    --model_name llama1b \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_leave_one_out

# bbh llama1b gs250 lr1e-4 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs250_lr1e-4_leaveoneout \
    --model_name llama1b \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_leave_one_out









# bbh llama1b gs5 lr1e-5 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs5_lr1e-5_leaveoneout \
    --model_name llama1b \
    --gs_iters 5 \
    --gs_lr 1e-5 \
    --gs_leave_one_out

# bbh llama1b gs25 lr1e-5 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs25_lr1e-5_leaveoneout \
    --model_name llama1b \
    --gs_iters 25 \
    --gs_lr 1e-5 \
    --gs_leave_one_out

# bbh llama1b gs100 lr1e-5 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs100_lr1e-5_leaveoneout \
    --model_name llama1b \
    --gs_iters 100 \
    --gs_lr 1e-5 \
    --gs_leave_one_out

# bbh llama1b gs250 lr1e-5 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs250_lr1e-5_leaveoneout \
    --model_name llama1b \
    --gs_iters 250 \
    --gs_lr 1e-5 \
    --gs_leave_one_out



# lr1e-2
# Submitted batch job 59161783 # 30.80
# Submitted batch job 59161784 # 29.49
# Submitted batch job 59161785 # 27.06
# Submitted batch job 59161786 # 27.98

# lr1e-3
# Submitted batch job 59161787 # 31.03
# Submitted batch job 59161788 # 32.37
# Submitted batch job 59161789 # 32.78
# Submitted batch job 59161790 # 32.65

# lr1e-4
# Submitted batch job 59161791 # 30.92
# Submitted batch job 59161792 # 31.03
# Submitted batch job 59161793 # 31.82
# Submitted batch job 59161794 # 33.01

# lr1e-5
# local 30.82
# local 31.03
# local 31.01
# local # likely not good anyway

# so far 33.01