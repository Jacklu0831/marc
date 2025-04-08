# python make_sbatch.py --ngpu 1 --time 1 --bash_files bash_cmds/0401_bbh/llama8b/0401_15_gs_numpermute.sh

# bbh llama8b gs5 lr1e-2 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr1e-2_permuten1024 \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 8

# bbh llama8b gs25 lr1e-2 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-2_permuten1024 \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 8

# bbh llama8b gs100 lr1e-2 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs100_lr1e-2_permuten1024 \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 8

# bbh llama8b gs250 lr1e-2 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs250_lr1e-2_permuten1024 \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 8






# bbh llama8b gs5 lr1e-3 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr1e-3_permuten1024 \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 8

# bbh llama8b gs25 lr1e-3 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-3_permuten1024 \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 8

# bbh llama8b gs100 lr1e-3 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs100_lr1e-3_permuten1024 \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 8

# bbh llama8b gs250 lr1e-3 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs250_lr1e-3_permuten1024 \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 8







# bbh llama8b gs5 lr1e-4 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr1e-4_permuten1024 \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 8

# bbh llama8b gs25 lr1e-4 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-4_permuten1024 \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 8

# bbh llama8b gs100 lr1e-4 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs100_lr1e-4_permuten1024 \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 8

# bbh llama8b gs250 lr1e-4 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs250_lr1e-4_permuten1024 \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 8



# Submitted batch job 59112486
# Submitted batch job 59112487
# Submitted batch job 59112488
# Submitted batch job 59112489

# Submitted batch job 59112490
# Submitted batch job 59112491
# Submitted batch job 59112492
# Submitted batch job 59112493

# Submitted batch job 59112494
# Submitted batch job 59112495
# Submitted batch job 59112496
# Submitted batch job 59112497