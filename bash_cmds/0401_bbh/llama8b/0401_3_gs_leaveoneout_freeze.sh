# python make_sbatch.py --ngpu 1 --time 6 --bash_files bash_cmds/0401_bbh/llama8b/0401_3_gs_leaveoneout_freeze.sh


# bbh llama8b gs4 lr1e-3 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs4_lr1e-3_leaveoneout_freeze \
    --model_name llama8b \
    --gs_iters 4 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out \
    --gs_freeze_instruct

# bbh llama8b gs6 lr1e-3 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs6_lr1e-3_leaveoneout_freeze \
    --model_name llama8b \
    --gs_iters 6 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out \
    --gs_freeze_instruct

# bbh llama8b gs8 lr1e-3 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs8_lr1e-3_leaveoneout_freeze \
    --model_name llama8b \
    --gs_iters 8 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out \
    --gs_freeze_instruct

# bbh llama8b gs10 lr1e-3 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs10_lr1e-3_leaveoneout_freeze \
    --model_name llama8b \
    --gs_iters 10 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out \
    --gs_freeze_instruct

# bbh llama8b gs12 lr1e-3 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs12_lr1e-3_leaveoneout_freeze \
    --model_name llama8b \
    --gs_iters 12 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_leave_one_out \
    --gs_freeze_instruct
