# python make_sbatch.py --ngpu 1 --time 6 --bash_files bash_cmds/0401_bbh/llama8b/0401_3_gs_leaveoneout_freeze.sh





# bbh llama8b gs5 lr1e-4 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr1e-4_leaveoneout_freeze \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out \
    --gs_freeze_instruct

# bbh llama8b gs25 lr1e-4 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-4_leaveoneout_freeze \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out \
    --gs_freeze_instruct

# bbh llama8b gs100 lr1e-4 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs100_lr1e-4_leaveoneout_freeze \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out \
    --gs_freeze_instruct

# bbh llama8b gs250 lr1e-4 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs250_lr1e-4_leaveoneout_freeze \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out \
    --gs_freeze_instruct






# bbh llama8b gs5 lr1e-5 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr1e-5_leaveoneout_freeze \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-5 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out \
    --gs_freeze_instruct

# bbh llama8b gs25 lr1e-5 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-5_leaveoneout_freeze \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-5 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out \
    --gs_freeze_instruct

# bbh llama8b gs100 lr1e-5 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs100_lr1e-5_leaveoneout_freeze \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-5 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out \
    --gs_freeze_instruct

# bbh llama8b gs250 lr1e-5 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs250_lr1e-5_leaveoneout_freeze \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-5 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out \
    --gs_freeze_instruct









# bbh llama8b gs5 lr1e-6 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr1e-6_leaveoneout_freeze \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-6 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out \
    --gs_freeze_instruct

# bbh llama8b gs25 lr1e-6 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-6_leaveoneout_freeze \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-6 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out \
    --gs_freeze_instruct

# bbh llama8b gs100 lr1e-6 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs100_lr1e-6_leaveoneout_freeze \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-6 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out \
    --gs_freeze_instruct

# bbh llama8b gs250 lr1e-6 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs250_lr1e-6_leaveoneout_freeze \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-6 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out \
    --gs_freeze_instruct



# lr1e-4

# lr1e-5

# lr1e-6