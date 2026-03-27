# python make_sbatch.py --ngpu 1 --time 1 --bash_files bash_cmds/0401_bbh/llama1b/0401_16_gs_lossoninput.sh

# bbh llama1b gs5 lr1e-2 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs5_lr1e-2_lossoninput \
    --model_name llama1b \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_loss_on_input

# bbh llama1b gs25 lr1e-2 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs25_lr1e-2_lossoninput \
    --model_name llama1b \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_loss_on_input

# bbh llama1b gs100 lr1e-2 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs100_lr1e-2_lossoninput \
    --model_name llama1b \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_loss_on_input

# bbh llama1b gs250 lr1e-2 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs250_lr1e-2_lossoninput \
    --model_name llama1b \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_loss_on_input






# bbh llama1b gs5 lr1e-3 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs5_lr1e-3_lossoninput \
    --model_name llama1b \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_loss_on_input

# bbh llama1b gs25 lr1e-3 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs25_lr1e-3_lossoninput \
    --model_name llama1b \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_loss_on_input

# bbh llama1b gs100 lr1e-3 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs100_lr1e-3_lossoninput \
    --model_name llama1b \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_loss_on_input

# bbh llama1b gs250 lr1e-3 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs250_lr1e-3_lossoninput \
    --model_name llama1b \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_loss_on_input







# bbh llama1b gs5 lr1e-4 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs5_lr1e-4_lossoninput \
    --model_name llama1b \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_loss_on_input

# bbh llama1b gs25 lr1e-4 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs25_lr1e-4_lossoninput \
    --model_name llama1b \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_loss_on_input

# bbh llama1b gs100 lr1e-4 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs100_lr1e-4_lossoninput \
    --model_name llama1b \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_loss_on_input

# bbh llama1b gs250 lr1e-4 lossoninput
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs250_lr1e-4_lossoninput \
    --model_name llama1b \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_loss_on_input



# Submitted batch job 59112519 # 29.78
# Submitted batch job 59112520 # 28.46
# Submitted batch job 59112521 # 27.75
# Submitted batch job 59112522 # 27.66

# Submitted batch job 59112523 # 31.98
# Submitted batch job 59112524 # 31.88
# Submitted batch job 59112525 # 30.93
# Submitted batch job 59112526 # 30.93

# Submitted batch job 59112527 # 31.11
# Submitted batch job 59112528 # 31.35
# Submitted batch job 59112529 # 31.98
# Submitted batch job 59112530 # 31.78

# so far 31.98