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


# Submitted batch job 59112471
# Submitted batch job 59112472
# Submitted batch job 59112473
# Submitted batch job 59112474

# Submitted batch job 59112475
# Submitted batch job 59112476
# Submitted batch job 59112477
# Submitted batch job 59112478

# Submitted batch job 59112479
# Submitted batch job 59112480
# Submitted batch job 59112481
# Submitted batch job 59112482