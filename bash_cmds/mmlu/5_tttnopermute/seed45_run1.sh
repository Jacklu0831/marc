# mmlu ttt iter5 seed45 nopermute run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter5_seed45_nopermute_run1 \
    --ttt_iters 5 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 1 \
    --seed 45

# mmlu ttt iter10 seed45 nopermute run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter10_seed45_nopermute_run1 \
    --ttt_iters 10 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 1 \
    --seed 45

# mmlu ttt iter20 seed45 nopermute run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter20_seed45_nopermute_run1 \
    --ttt_iters 20 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 1 \
    --seed 45

# mmlu ttt iter30 seed45 nopermute run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter30_seed45_nopermute_run1 \
    --ttt_iters 30 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 1 \
    --seed 45

# 44.196522344024615
# 45.06386031898469
# 43.402329763916015
# 41.31216632047929