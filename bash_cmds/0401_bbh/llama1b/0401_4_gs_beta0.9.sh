# python make_sbatch.py --ngpu 1 --time 1 --bash_files bash_cmds/0401_bbh/llama1b/0401_4_gs_beta0.9.sh

# bbh llama1b gs5 lr1e-2 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama1b_gs5_lr1e-2_beta0.9 \
    --model_name llama1b \
    --gs_iters 5 \
    --gs_lr 1e-2 \
    --gs_beta2 0.9

# bbh llama1b gs25 lr1e-2 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama1b_gs25_lr1e-2_beta0.9 \
    --model_name llama1b \
    --gs_iters 25 \
    --gs_lr 1e-2 \
    --gs_beta2 0.9

# bbh llama1b gs100 lr1e-2 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama1b_gs100_lr1e-2_beta0.9 \
    --model_name llama1b \
    --gs_iters 100 \
    --gs_lr 1e-2 \
    --gs_beta2 0.9

# bbh llama1b gs250 lr1e-2 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama1b_gs250_lr1e-2_beta0.9 \
    --model_name llama1b \
    --gs_iters 250 \
    --gs_lr 1e-2 \
    --gs_beta2 0.9






# bbh llama1b gs5 lr1e-3 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama1b_gs5_lr1e-3_beta0.9 \
    --model_name llama1b \
    --gs_iters 5 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9

# bbh llama1b gs25 lr1e-3 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama1b_gs25_lr1e-3_beta0.9 \
    --model_name llama1b \
    --gs_iters 25 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9

# bbh llama1b gs100 lr1e-3 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama1b_gs100_lr1e-3_beta0.9 \
    --model_name llama1b \
    --gs_iters 100 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9

# bbh llama1b gs250 lr1e-3 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama1b_gs250_lr1e-3_beta0.9 \
    --model_name llama1b \
    --gs_iters 250 \
    --gs_lr 1e-3 \
    --gs_beta2 0.9







# bbh llama1b gs5 lr1e-4 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama1b_gs5_lr1e-4_beta0.9 \
    --model_name llama1b \
    --gs_iters 5 \
    --gs_lr 1e-4 \
    --gs_beta2 0.9

# bbh llama1b gs25 lr1e-4 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama1b_gs25_lr1e-4_beta0.9 \
    --model_name llama1b \
    --gs_iters 25 \
    --gs_lr 1e-4 \
    --gs_beta2 0.9

# bbh llama1b gs100 lr1e-4 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama1b_gs100_lr1e-4_beta0.9 \
    --model_name llama1b \
    --gs_iters 100 \
    --gs_lr 1e-4 \
    --gs_beta2 0.9

# bbh llama1b gs250 lr1e-4 beta0.9
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --flash_attn \
    --tag bbh_llama1b_gs250_lr1e-4_beta0.9 \
    --model_name llama1b \
    --gs_iters 250 \
    --gs_lr 1e-4 \
    --gs_beta2 0.9



# Submitted batch job 59111896 # 28.54
# Submitted batch job 59111897 # 24.27
# Submitted batch job 59111898 # 23.62
# Submitted batch job 59111899 # 21.52

# Submitted batch job 59111900 # 32.03
# Submitted batch job 59111901 # 31.81
# Submitted batch job 59111902 # 30.33
# Submitted batch job 59111903 # 29.08

# Submitted batch job 59111904 # 31.25
# Submitted batch job 59111905 # 31.59
# Submitted batch job 59111906 # 32.68
# Submitted batch job 59111907 # 32.44

# so far 32.68