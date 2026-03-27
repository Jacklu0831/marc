# python make_sbatch.py --ngpu 1 --time 1 --bash_files bash_cmds/0401_bbh/llama1b/0401_17_gs_demon5.sh

# bbh llama1b gs5 lr1e-2 demon5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs5_lr1e-2_demon5 \
    --model_name llama1b \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --num_demonstrations 5

# bbh llama1b gs25 lr1e-2 demon5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs25_lr1e-2_demon5 \
    --model_name llama1b \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --num_demonstrations 5

# bbh llama1b gs100 lr1e-2 demon5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs100_lr1e-2_demon5 \
    --model_name llama1b \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --num_demonstrations 5

# bbh llama1b gs250 lr1e-2 demon5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs250_lr1e-2_demon5 \
    --model_name llama1b \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --num_demonstrations 5






# bbh llama1b gs5 lr1e-3 demon5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs5_lr1e-3_demon5 \
    --model_name llama1b \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --num_demonstrations 5

# bbh llama1b gs25 lr1e-3 demon5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs25_lr1e-3_demon5 \
    --model_name llama1b \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --num_demonstrations 5

# bbh llama1b gs100 lr1e-3 demon5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs100_lr1e-3_demon5 \
    --model_name llama1b \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --num_demonstrations 5

# bbh llama1b gs250 lr1e-3 demon5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs250_lr1e-3_demon5 \
    --model_name llama1b \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --num_demonstrations 5







# bbh llama1b gs5 lr1e-4 demon5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs5_lr1e-4_demon5 \
    --model_name llama1b \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --num_demonstrations 5

# bbh llama1b gs25 lr1e-4 demon5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs25_lr1e-4_demon5 \
    --model_name llama1b \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --num_demonstrations 5

# bbh llama1b gs100 lr1e-4 demon5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs100_lr1e-4_demon5 \
    --model_name llama1b \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --num_demonstrations 5

# bbh llama1b gs250 lr1e-4 demon5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama1b_gs250_lr1e-4_demon5 \
    --model_name llama1b \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --num_demonstrations 5


# lr1e-2
# Submitted batch job 59163567 # 28.49
# Submitted batch job 59163568 # 24.04
# Submitted batch job 59163569 # 25.21
# Submitted batch job 59163570 # 24.77

# lr1e-3
# Submitted batch job 59163571 # 31.61
# Submitted batch job 59163572 # 30.65
# Submitted batch job 59163573 # 30.55
# Submitted batch job 59163574 # 30.46

# lr1e-4
# Submitted batch job 59163575 # 32.03
# Submitted batch job 59163576 # 33.00
# Submitted batch job 59163577 # 31.52
# Submitted batch job 59163578 # 31.21