# mmlu ttt iter10 save seed43 run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter10_save_seed43_run1 \
    --ttt_iters 10 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 400 \
    --ttt_save \
    --seed 43

# mmlu ttt iter20 save seed43 run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter20_save_seed43_run1 \
    --ttt_iters 20 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 400 \
    --ttt_save \
    --seed 43

# mmlu ttt iter30 save seed43 run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter30_save_seed43_run1 \
    --ttt_iters 30 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 400 \
    --ttt_save \
    --seed 43

# mmlu ttt iter40 save seed43 run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter40_save_seed43_run1 \
    --ttt_iters 40 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 400 \
    --ttt_save \
    --seed 43

# mmlu ttt iter50 save seed43 run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter50_save_seed43_run1 \
    --ttt_iters 50 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 400 \
    --ttt_save \
    --seed 43

# 44.36091325766655
# 43.20266958426503
# 39.641111986309625
# 39.31634388281754
# 38.24482474069486