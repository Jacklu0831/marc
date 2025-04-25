# python make_sbatch.py --ngpu 1 --time 12 --single --bash_files bash_cmds/0401_bbh/llama8b/0420_0_evalondemon.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)






# bbh llama8b evalondemon
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_evalondemon \
    --model_name llama8b \
    --eval_on_demonstrations





# bbh llama8b ttt iter5 evalondemon
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter5_evalondemon \
    --model_name llama8b \
    --ttt_iters 5 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 5000 \
    --eval_on_demonstrations

# bbh llama8b ttt iter10 evalondemon
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter10_evalondemon \
    --model_name llama8b \
    --ttt_iters 10 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 5000 \
    --eval_on_demonstrations

# bbh llama8b ttt iter15 evalondemon
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter15_evalondemon \
    --model_name llama8b \
    --ttt_iters 15 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 5000 \
    --eval_on_demonstrations

# bbh llama8b ttt iter20 evalondemon
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter20_evalondemon \
    --model_name llama8b \
    --ttt_iters 20 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 5000 \
    --eval_on_demonstrations

# bbh llama8b ttt iter25 evalondemon
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_llama8b_ttt_iter25_evalondemon \
    --model_name llama8b \
    --ttt_iters 25 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 5000 \
    --eval_on_demonstrations

# Submitted batch job 59480603

# 83.33333333333333
# 90.74074074074075
# 94.81481481481481
# 94.07407407407408
# 87.77777777777777
# 83.70370370370371