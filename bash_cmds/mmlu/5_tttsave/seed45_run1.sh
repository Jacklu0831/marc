# mmlu ttt iter10 save seed45 run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter10_save_seed45_run1 \
    --ttt_iters 10 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 400 \
    --ttt_save \
    --seed 45

# mmlu ttt iter20 save seed45 run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter20_save_seed45_run1 \
    --ttt_iters 20 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 400 \
    --ttt_save \
    --seed 45

# mmlu ttt iter30 save seed45 run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter30_save_seed45_run1 \
    --ttt_iters 30 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 400 \
    --ttt_save \
    --seed 45

# mmlu ttt iter40 save seed45 run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter40_save_seed45_run1 \
    --ttt_iters 40 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 400 \
    --ttt_save \
    --seed 45

# mmlu ttt iter50 save seed45 run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter50_save_seed45_run1 \
    --ttt_iters 50 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 400 \
    --ttt_save \
    --seed 45

# 46.03222865572633
# 44.240207939408414
# 40.632436083926045
# 39.531143068767605
# 38.627304237490186