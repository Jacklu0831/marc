# python make_sbatch.py --ngpu 1 --time 3 --bash_files bash_cmds/0401_bbh/llama8b/0401_15_gs_numpermute_rerun.sh





# bbh llama8b gs5 lr1e-3 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr1e-3_permuten1024_rerun \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 4

# bbh llama8b gs25 lr1e-3 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-3_permuten1024_rerun \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 4

# bbh llama8b gs100 lr1e-3 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs100_lr1e-3_permuten1024_rerun \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 4

# bbh llama8b gs250 lr1e-3 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs250_lr1e-3_permuten1024_rerun \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 4











# bbh llama8b gs5 lr3e-4 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr3e-4_permuten1024_rerun \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 3e-4 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 4

# bbh llama8b gs25 lr3e-4 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr3e-4_permuten1024_rerun \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 3e-4 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 4

# bbh llama8b gs100 lr3e-4 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs100_lr3e-4_permuten1024_rerun \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 3e-4 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 4

# bbh llama8b gs250 lr3e-4 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs250_lr3e-4_permuten1024_rerun \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 3e-4 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 4









# bbh llama8b gs5 lr1e-4 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs5_lr1e-4_permuten1024_rerun \
    --model_name llama8b \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 4

# bbh llama8b gs25 lr1e-4 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs25_lr1e-4_permuten1024_rerun \
    --model_name llama8b \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 4

# bbh llama8b gs100 lr1e-4 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs100_lr1e-4_permuten1024_rerun \
    --model_name llama8b \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 4

# bbh llama8b gs250 lr1e-4 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_gs250_lr1e-4_permuten1024_rerun \
    --model_name llama8b \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024 \
    --gs_batch_size 2 \
    --gs_grad_accum_steps 5 \
    --gs_permute_batch_size 4



# lr1e-3
# Submitted batch job 59266409 # 43.25
# Submitted batch job 59266410 # 50.87
# Submitted batch job 59266411 # 49.98
# Submitted batch job 59266412 # 49.04

# lr3e-4
# Submitted batch job 59266413 # 42.04
# Submitted batch job 59266414 # 48.21
# Submitted batch job 59266415 # 50.17
# Submitted batch job 59266416 # 50.29

# lr1e-4
# Submitted batch job 59266417 # 40.42
# Submitted batch job 59266418 # 41.85
# Submitted batch job 59266419 # 50.23
# Submitted batch job 59266420 # 50.69
