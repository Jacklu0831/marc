# python make_sbatch.py --ngpu 1 --time 2 --bash_files bash_cmds/0401_bbh/llama1b/0401_15_gs_numpermute.sh

# bbh llama1b gs5 lr1e-2 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs5_lr1e-2_permuten1024 \
    --model_name llama1b \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024

# bbh llama1b gs25 lr1e-2 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs25_lr1e-2_permuten1024 \
    --model_name llama1b \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024

# bbh llama1b gs100 lr1e-2 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs100_lr1e-2_permuten1024 \
    --model_name llama1b \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024

# bbh llama1b gs250 lr1e-2 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs250_lr1e-2_permuten1024 \
    --model_name llama1b \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_num_permute 1024






# bbh llama1b gs5 lr1e-3 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs5_lr1e-3_permuten1024 \
    --model_name llama1b \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024

# bbh llama1b gs25 lr1e-3 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs25_lr1e-3_permuten1024 \
    --model_name llama1b \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024

# bbh llama1b gs100 lr1e-3 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs100_lr1e-3_permuten1024 \
    --model_name llama1b \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024

# bbh llama1b gs250 lr1e-3 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs250_lr1e-3_permuten1024 \
    --model_name llama1b \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_num_permute 1024







# bbh llama1b gs5 lr1e-4 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs5_lr1e-4_permuten1024 \
    --model_name llama1b \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024

# bbh llama1b gs25 lr1e-4 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs25_lr1e-4_permuten1024 \
    --model_name llama1b \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024

# bbh llama1b gs100 lr1e-4 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs100_lr1e-4_permuten1024 \
    --model_name llama1b \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024

# bbh llama1b gs250 lr1e-4 permuten1024
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs250_lr1e-4_permuten1024 \
    --model_name llama1b \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_num_permute 1024


# Submitted batch job 59112471 # 31.91
# Submitted batch job 59112472 # 34.83
# Submitted batch job 59112473 # 30.24
# Submitted batch job 59112474 # 28.94

# Submitted batch job 59112475 # 29.73
# Submitted batch job 59112476 # 35.45
# Submitted batch job 59112477 # 40.41
# Submitted batch job 59112478 # 40.28

# Submitted batch job 59112479 # 27.39
# Submitted batch job 59112480 # 29.37
# Submitted batch job 59112481 # 32.07
# Submitted batch job 59112482 # 38.83

# so far 40.41, but why? debug this




# lr1e-2
# Submitted batch job 59139790
# Submitted batch job 59139791
# Submitted batch job 59139792
# Submitted batch job 59139793

# lr1e-3
# Submitted batch job 59139794
# Submitted batch job 59139795
# Submitted batch job 59139796
# Submitted batch job 59139797

# lr1e-4
# Submitted batch job 59139798
# Submitted batch job 59139799
# Submitted batch job 59139800
# Submitted batch job 59139801