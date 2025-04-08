# python make_sbatch.py --ngpu 1 --time 1 --bash_files bash_cmds/0401_bbh/llama1b/0401_2_gs.sh

# bbh llama1b gs5 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs5_lr1e-2 \
    --model_name llama1b \
    --gs_iters 5 \
    --gs_lr 1e-2

# bbh llama1b gs25 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs25_lr1e-2 \
    --model_name llama1b \
    --gs_iters 25 \
    --gs_lr 1e-2

# bbh llama1b gs100 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs100_lr1e-2 \
    --model_name llama1b \
    --gs_iters 100 \
    --gs_lr 1e-2

# bbh llama1b gs250 lr1e-2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs250_lr1e-2 \
    --model_name llama1b \
    --gs_iters 250 \
    --gs_lr 1e-2






# bbh llama1b gs5 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs5_lr1e-3 \
    --model_name llama1b \
    --gs_iters 5 \
    --gs_lr 1e-3

# bbh llama1b gs25 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs25_lr1e-3 \
    --model_name llama1b \
    --gs_iters 25 \
    --gs_lr 1e-3

# bbh llama1b gs100 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs100_lr1e-3 \
    --model_name llama1b \
    --gs_iters 100 \
    --gs_lr 1e-3

# bbh llama1b gs250 lr1e-3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs250_lr1e-3 \
    --model_name llama1b \
    --gs_iters 250 \
    --gs_lr 1e-3







# bbh llama1b gs5 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs5_lr1e-4 \
    --model_name llama1b \
    --gs_iters 5 \
    --gs_lr 1e-4

# bbh llama1b gs25 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs25_lr1e-4 \
    --model_name llama1b \
    --gs_iters 25 \
    --gs_lr 1e-4

# bbh llama1b gs100 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs100_lr1e-4 \
    --model_name llama1b \
    --gs_iters 100 \
    --gs_lr 1e-4

# bbh llama1b gs250 lr1e-4
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs250_lr1e-4 \
    --model_name llama1b \
    --gs_iters 250 \
    --gs_lr 1e-4


# Submitted batch job 59111720
# Submitted batch job 59111721
# Submitted batch job 59111722
# Submitted batch job 59111723

# Submitted batch job 59111724
# Submitted batch job 59111725
# Submitted batch job 59111726
# Submitted batch job 59111727

# Submitted batch job 59111728
# Submitted batch job 59111729
# Submitted batch job 59111730
# Submitted batch job 59111731