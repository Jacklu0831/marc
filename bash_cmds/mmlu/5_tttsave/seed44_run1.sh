# mmlu ttt iter10 save seed44 run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter10_save_seed44_run1 \
    --ttt_iters 10 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 400 \
    --ttt_save \
    --seed 44

# mmlu ttt iter20 save seed44 run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter20_save_seed44_run1 \
    --ttt_iters 20 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 400 \
    --ttt_save \
    --seed 44

# mmlu ttt iter30 save seed44 run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter30_save_seed44_run1 \
    --ttt_iters 30 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 400 \
    --ttt_save \
    --seed 44

# mmlu ttt iter40 save seed44 run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter40_save_seed44_run1 \
    --ttt_iters 40 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 400 \
    --ttt_save \
    --seed 44

# mmlu ttt iter50 save seed44 run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_ttt_iter50_save_seed44_run1 \
    --ttt_iters 50 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 400 \
    --ttt_save \
    --seed 44

# 45.53995714191352
# 42.96896744535117
# 40.75757947170582
# 38.20658657022649
# 37.56019668180568