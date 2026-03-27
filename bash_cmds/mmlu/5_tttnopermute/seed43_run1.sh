# mmlu ttt iter5 seed43 nopermute run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter5_seed43_nopermute_run1 \
    --ttt_iters 5 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 1 \
    --seed 43

# mmlu ttt iter10 seed43 nopermute run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter10_seed43_nopermute_run1 \
    --ttt_iters 10 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 1 \
    --seed 43

# mmlu ttt iter20 seed43 nopermute run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter20_seed43_nopermute_run1 \
    --ttt_iters 20 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 1 \
    --seed 43

# mmlu ttt iter30 seed43 nopermute run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter30_seed43_nopermute_run1 \
    --ttt_iters 30 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 1 \
    --seed 43

# 42.71982439489021
# 43.281398418964315
# 43.21048361143845
# 41.98741962142265