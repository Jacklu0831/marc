# python make_sbatch.py --ngpu 1 --time 9 --bash_files bash_cmds/0401_bbh/llama8b/0401_15_gs_numpermute.sh





# bbh llama8b gs5 lr1e-3 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr1e-3_permuten1024 \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 4

# bbh llama8b gs25 lr1e-3 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-3_permuten1024 \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 4

# bbh llama8b gs100 lr1e-3 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs100_lr1e-3_permuten1024 \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 4

# bbh llama8b gs250 lr1e-3 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs250_lr1e-3_permuten1024 \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 4







# bbh llama8b gs5 lr1e-4 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr1e-4_permuten1024 \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 4

# bbh llama8b gs25 lr1e-4 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-4_permuten1024 \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 4

# bbh llama8b gs100 lr1e-4 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs100_lr1e-4_permuten1024 \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 4

# bbh llama8b gs250 lr1e-4 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs250_lr1e-4_permuten1024 \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 4






# bbh llama8b gs5 lr1e-5 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr1e-5_permuten1024 \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-5 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 4

# bbh llama8b gs25 lr1e-5 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-5_permuten1024 \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-5 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 4

# bbh llama8b gs100 lr1e-5 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs100_lr1e-5_permuten1024 \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-5 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 4

# bbh llama8b gs250 lr1e-5 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs250_lr1e-5_permuten1024 \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-5 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 4



# lr1e-3
# Submitted batch job 59219135 # 43.18
# Submitted batch job 59219136 # 51.03
# Submitted batch job 59219137 # 49.67
# Submitted batch job 59219138 # 48.76

# lr1e-4
# Submitted batch job 59219139 # 40.62
# Submitted batch job 59219140 # 41.85
# Submitted batch job 59219141 # 50.12
# Submitted batch job 59219142 # 50.94

# lr1e-5
# Submitted batch job 59219143 # 40.12
# Submitted batch job 59219144 # 40.35
# Submitted batch job 59219145 # 41.33
# Submitted batch job 59219146 # 42.31

# so far 51.03, oh no its not transferring
# it likely works for llama1b because other runs overfit, but here its not probably due to sensitivity to hyperparam?