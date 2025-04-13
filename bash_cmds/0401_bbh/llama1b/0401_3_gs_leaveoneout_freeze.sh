# python make_sbatch.py --ngpu 1 --time 2 --bash_files bash_cmds/0401_bbh/llama1b/0401_3_gs_leaveoneout_freeze.sh



# bbh llama1b gs5 lr1e-2 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs5_lr1e-2_leaveoneout_freeze \
    --model_name llama1b \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_leave_one_out \
    --gs_freeze_instruct

# bbh llama1b gs25 lr1e-2 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs25_lr1e-2_leaveoneout_freeze \
    --model_name llama1b \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_leave_one_out \
    --gs_freeze_instruct

# bbh llama1b gs100 lr1e-2 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs100_lr1e-2_leaveoneout_freeze \
    --model_name llama1b \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_leave_one_out \
    --gs_freeze_instruct

# bbh llama1b gs250 lr1e-2 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs250_lr1e-2_leaveoneout_freeze \
    --model_name llama1b \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_leave_one_out \
    --gs_freeze_instruct






# bbh llama1b gs5 lr1e-3 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs5_lr1e-3_leaveoneout_freeze \
    --model_name llama1b \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_freeze_instruct

# bbh llama1b gs25 lr1e-3 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs25_lr1e-3_leaveoneout_freeze \
    --model_name llama1b \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_freeze_instruct

# bbh llama1b gs100 lr1e-3 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs100_lr1e-3_leaveoneout_freeze \
    --model_name llama1b \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_freeze_instruct

# bbh llama1b gs250 lr1e-3 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs250_lr1e-3_leaveoneout_freeze \
    --model_name llama1b \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_leave_one_out \
    --gs_freeze_instruct







# bbh llama1b gs5 lr1e-4 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs5_lr1e-4_leaveoneout_freeze \
    --model_name llama1b \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --gs_freeze_instruct

# bbh llama1b gs25 lr1e-4 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs25_lr1e-4_leaveoneout_freeze \
    --model_name llama1b \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --gs_freeze_instruct

# bbh llama1b gs100 lr1e-4 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs100_lr1e-4_leaveoneout_freeze \
    --model_name llama1b \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --gs_freeze_instruct

# bbh llama1b gs250 lr1e-4 leaveoneout freeze
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs250_lr1e-4_leaveoneout_freeze \
    --model_name llama1b \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_leave_one_out \
    --gs_freeze_instruct



# lr1e-2
# Submitted batch job 59216496 # 29.92
# Submitted batch job 59216497 # 29.88
# Submitted batch job 59216498 # 30.12
# Submitted batch job 59216499 # 29.97

# lr1e-3
# Submitted batch job 59216500 # 30.87
# Submitted batch job 59216501 # 31.04
# Submitted batch job 59216502 # 31.32
# Submitted batch job 59216503 # 31.27

# lr1e-4
# Submitted batch job 59216504 # 30.96
# Submitted batch job 59216505 # 30.75
# Submitted batch job 59216506 # 30.90
# Submitted batch job 59216507 # 30.65