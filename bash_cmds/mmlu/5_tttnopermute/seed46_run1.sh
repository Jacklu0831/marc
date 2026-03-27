# mmlu ttt iter5 seed46 nopermute run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter5_seed46_nopermute_run1 \
    --ttt_iters 5 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 1 \
    --seed 46

# mmlu ttt iter10 seed46 nopermute run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter10_seed46_nopermute_run1 \
    --ttt_iters 10 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 1 \
    --seed 46

# mmlu ttt iter20 seed46 nopermute run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter20_seed46_nopermute_run1 \
    --ttt_iters 20 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 1 \
    --seed 46

# mmlu ttt iter30 seed46 nopermute run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter30_seed46_nopermute_run1 \
    --ttt_iters 30 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 1 \
    --seed 46

# 43.85820769543362
# 44.64796504671198
# 42.11723436661303
# 38.74894976566541