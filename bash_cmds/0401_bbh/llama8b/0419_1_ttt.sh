# python make_sbatch.py --ngpu 1 --time 12 --single --bash_files bash_cmds/0401_bbh/llama8b/0419_1_ttt.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)









# bbh llama8b ttt iter5
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter5 \
    --model_name llama8b \
    --ttt_iters 5 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 5000

# bbh llama8b ttt iter10
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter10 \
    --model_name llama8b \
    --ttt_iters 10 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 5000

# bbh llama8b ttt iter15
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter15 \
    --model_name llama8b \
    --ttt_iters 15 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 5000

# bbh llama8b ttt iter20
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter20 \
    --model_name llama8b \
    --ttt_iters 20 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 5000

# bbh llama8b ttt iter25
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter25 \
    --model_name llama8b \
    --ttt_iters 25 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 5000

# Submitted batch job 59478567

# 53.73449536751297
# 55.29418358680572
# 51.803370751941706
# 47.02400514505865
# 42.63207291705132

# so far 55.29418358680572