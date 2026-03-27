# python make_sbatch.py --ngpu 1 --time 6 --bash_files bash_cmds/0401_bbh/llama8b/0401_2_gs_leaveoneout.sh

# bbh llama8b gs5 lr1e-2 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr1e-2_leaveoneout \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out

# bbh llama8b gs25 lr1e-2 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-2_leaveoneout \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out

# bbh llama8b gs100 lr1e-2 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs100_lr1e-2_leaveoneout \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out

# bbh llama8b gs250 lr1e-2 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs250_lr1e-2_leaveoneout \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out






# bbh llama8b gs5 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr1e-3_leaveoneout \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out

# bbh llama8b gs25 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-3_leaveoneout \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out

# bbh llama8b gs100 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs100_lr1e-3_leaveoneout \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out

# bbh llama8b gs250 lr1e-3 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs250_lr1e-3_leaveoneout \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out







# bbh llama8b gs5 lr1e-4 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr1e-4_leaveoneout \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out

# bbh llama8b gs25 lr1e-4 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-4_leaveoneout \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out

# bbh llama8b gs100 lr1e-4 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs100_lr1e-4_leaveoneout \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out

# bbh llama8b gs250 lr1e-4 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs250_lr1e-4_leaveoneout \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out






# bbh llama8b gs5 lr1e-5 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr1e-5_leaveoneout \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-5 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out

# bbh llama8b gs25 lr1e-5 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-5_leaveoneout \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-5 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out

# bbh llama8b gs100 lr1e-5 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs100_lr1e-5_leaveoneout \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-5 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out

# bbh llama8b gs250 lr1e-5 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs250_lr1e-5_leaveoneout \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-5 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out









# bbh llama8b gs5 lr1e-6 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr1e-6_leaveoneout \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-6 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out

# bbh llama8b gs25 lr1e-6 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-6_leaveoneout \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-6 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out

# bbh llama8b gs100 lr1e-6 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs100_lr1e-6_leaveoneout \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-6 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out

# bbh llama8b gs250 lr1e-6 leaveoneout
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs250_lr1e-6_leaveoneout \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-6 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out





# lr1e-2
# Submitted batch job 59161808 # 39.64
# Submitted batch job 59161809 # 34.44
# Submitted batch job 59161810 # 27.00
# Submitted batch job 59161811 # 25.51

# lr1e-3
# Submitted batch job 59161812 # 47.47
# Submitted batch job 59161813 # 49.14
# Submitted batch job 59161814 # 47.87
# Submitted batch job 59161815 # 47.54

# lr1e-4
# Submitted batch job 59161816 # 49.05
# Submitted batch job 59161817 # 48.72
# Submitted batch job 59161818 # 48.78
# Submitted batch job 59161819 # 49.24

# lr1e-5
# Submitted batch job 59161820 # 48.95
# Submitted batch job 59161821 # 49.01
# Submitted batch job 59161822 # 49.05
# Submitted batch job 59161823 # 48.92

# lr1e-6
# Submitted batch job 59161824 # 49.17
# Submitted batch job 59161825 # 49.09
# Submitted batch job 59161826 # 49.29
# Submitted batch job 59161827 # 49.17

# so far 49.29