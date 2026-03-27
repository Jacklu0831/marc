# mmlu ttt iter15 save seed42 run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag mmlu_ttt_iter15_save_seed42_run1 \
    --ttt_iters 15 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 250 \
    --ttt_save \
    --seed 42

# mmlu ttt iter15 save seed43 run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag mmlu_ttt_iter15_save_seed43_run1 \
    --ttt_iters 15 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 250 \
    --ttt_save \
    --seed 43

# mmlu ttt iter15 save seed44 run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag mmlu_ttt_iter15_save_seed44_run1 \
    --ttt_iters 15 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 250 \
    --ttt_save \
    --seed 44

# mmlu ttt iter15 save seed45 run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag mmlu_ttt_iter15_save_seed45_run1 \
    --ttt_iters 15 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 250 \
    --ttt_save \
    --seed 45

# mmlu ttt iter15 save seed46 run1
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag mmlu_ttt_iter15_save_seed46_run1 \
    --ttt_iters 15 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 250 \
    --ttt_save \
    --seed 46

# 47.110472126340724
# 45.75473543977815
# 42.74036349404544
# 46.074319705403866
# 44.4853839897683