# mmlu ttt iter15 save seed42 run2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag mmlu_ttt_iter15_save_seed42_run2 \
    --ttt_iters 15 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 250 \
    --ttt_save \
    --seed 42

# mmlu ttt iter15 save seed43 run2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag mmlu_ttt_iter15_save_seed43_run2 \
    --ttt_iters 15 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 250 \
    --ttt_save \
    --seed 43

# mmlu ttt iter15 save seed44 run2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag mmlu_ttt_iter15_save_seed44_run2 \
    --ttt_iters 15 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 250 \
    --ttt_save \
    --seed 44

# mmlu ttt iter15 save seed45 run2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag mmlu_ttt_iter15_save_seed45_run2 \
    --ttt_iters 15 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 250 \
    --ttt_save \
    --seed 45

# mmlu ttt iter15 save seed46 run2
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag mmlu_ttt_iter15_save_seed46_run2 \
    --ttt_iters 15 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 250 \
    --ttt_save \
    --seed 46

# 47.06323865805462
# 45.735023237925205
# 42.64218094617249
# 46.05241035686913
# 44.50509619162125