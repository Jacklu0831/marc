# python make_sbatch.py --ngpu 1 --time 6 --bash_files bash_cmds/0401_bbh/llama8b/0401_4_gs_beta0.9.sh

# bbh llama8b gs5 lr1e-2 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs5_lr1e-2_beta0.9 \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_beta2 0.9 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# bbh llama8b gs25 lr1e-2 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs25_lr1e-2_beta0.9 \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_beta2 0.9 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# bbh llama8b gs100 lr1e-2 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs100_lr1e-2_beta0.9 \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_beta2 0.9 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# bbh llama8b gs250 lr1e-2 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs250_lr1e-2_beta0.9 \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_beta2 0.9 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5






# bbh llama8b gs5 lr1e-3 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs5_lr1e-3_beta0.9 \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# bbh llama8b gs25 lr1e-3 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs25_lr1e-3_beta0.9 \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# bbh llama8b gs100 lr1e-3 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs100_lr1e-3_beta0.9 \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# bbh llama8b gs250 lr1e-3 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs250_lr1e-3_beta0.9 \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5







# bbh llama8b gs5 lr1e-4 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs5_lr1e-4_beta0.9 \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_beta2 0.9 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# bbh llama8b gs25 lr1e-4 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs25_lr1e-4_beta0.9 \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_beta2 0.9 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# bbh llama8b gs100 lr1e-4 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs100_lr1e-4_beta0.9 \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_beta2 0.9 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5

# bbh llama8b gs250 lr1e-4 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama8b_gs250_lr1e-4_beta0.9 \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_beta2 0.9 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5



# Submitted batch job 59111915 # 30.43
# Submitted batch job 59111916 # 21.86
# Submitted batch job 59111917 # 21.77
# Submitted batch job 59111918 # 24.79

# Submitted batch job 59111919 # 48.74
# Submitted batch job 59111920 # 46.67
# Submitted batch job 59111921 # 44.23
# Submitted batch job 59111922 # 42.33

# Submitted batch job 59111923 # 49.32
# Submitted batch job 59111924 # 48.37
# Submitted batch job 59111925 # 49.00
# Submitted batch job 59111926 # 49.23

# so far 49.32