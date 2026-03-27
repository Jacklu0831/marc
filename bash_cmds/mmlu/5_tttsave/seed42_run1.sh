# mmlu ttt iter10 save seed42 run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter10_save_seed42_run1 \
    --ttt_iters 10 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 400 \
    --ttt_save \
    --seed 42

# mmlu ttt iter20 save seed42 run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter20_save_seed42_run1 \
    --ttt_iters 20 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 400 \
    --ttt_save \
    --seed 42

# mmlu ttt iter30 save seed42 run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter30_save_seed42_run1 \
    --ttt_iters 30 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 400 \
    --ttt_save \
    --seed 42

# mmlu ttt iter40 save seed42 run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter40_save_seed42_run1 \
    --ttt_iters 40 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 400 \
    --ttt_save \
    --seed 42

# mmlu ttt iter50 save seed42 run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter50_save_seed42_run1 \
    --ttt_iters 50 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 400 \
    --ttt_save \
    --seed 42

# 44.10397735727034
# 43.60028105255887
# 41.078978771578434
# 39.75721045485448
# 38.98809010098473