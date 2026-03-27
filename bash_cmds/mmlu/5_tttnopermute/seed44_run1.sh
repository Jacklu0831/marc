# mmlu ttt iter5 seed44 nopermute run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter5_seed44_nopermute_run1 \
    --ttt_iters 5 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 1 \
    --seed 44

# mmlu ttt iter10 seed44 nopermute run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter10_seed44_nopermute_run1 \
    --ttt_iters 10 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 1 \
    --seed 44

# mmlu ttt iter20 seed44 nopermute run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter20_seed44_nopermute_run1 \
    --ttt_iters 20 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 1 \
    --seed 44

# mmlu ttt iter30 seed44 nopermute run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter30_seed44_nopermute_run1 \
    --ttt_iters 30 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 1 \
    --seed 44

# 44.78115807705461
# 44.51812241736569
# 42.50097574395508
# 40.78967567525282