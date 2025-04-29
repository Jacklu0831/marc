# python make_sbatch.py --ngpu 1 --time 4 --single --bash_files bash_cmds/0401_mmlu/0401_tttsearch/0.sh
MASTER_PORT=$(comm -23 <(seq 10000 65000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)





# mmlu ttt iter10
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter10 \
    --ttt_iters 10 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 250

# mmlu ttt iter20
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter20 \
    --ttt_iters 20 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 250

# mmlu ttt iter30
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter30 \
    --ttt_iters 30 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 250

# mmlu ttt iter40
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter40 \
    --ttt_iters 40 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 250

# mmlu ttt iter50
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter50 \
    --ttt_iters 50 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 250

# 44.56372777652913
# 43.548040811381455
# 42.07776789658206
# 39.05165340789101
# 38.18877071527434