# python make_sbatch.py --ngpu 1 --time 12 --bash_files bash_cmds/0401_bbh/llama8b/0401_15_gs_numpermute.sh

# bbh llama8b gs5 lr1e-2 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr1e-2_permuten1024 \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 4

# bbh llama8b gs25 lr1e-2 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-2_permuten1024 \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 4

# bbh llama8b gs100 lr1e-2 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs100_lr1e-2_permuten1024 \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 4

# bbh llama8b gs250 lr1e-2 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs250_lr1e-2_permuten1024 \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 4






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




# lr1e-2
# Submitted batch job 59125453
# Submitted batch job 59125454
# Submitted batch job 59125455
# Submitted batch job 59125456

# lr1e-3
# Submitted batch job 59125457
# Submitted batch job 59125458
# Submitted batch job 59125459
# Submitted batch job 59125460

# lr1e-4
# Submitted batch job 59125461
# Submitted batch job 59125462
# Submitted batch job 59125463
# Submitted batch job 59125464

# lr1e-5
# Submitted batch job 59131327
# Submitted batch job 59131328
# Submitted batch job 59131329
# Submitted batch job 59131330