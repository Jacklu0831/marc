# mmlu ttt iter15 save seed42 run3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag mmlu_ttt_iter15_save_seed42_run3 \
    --ttt_iters 15 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 250 \
    --ttt_save \
    --seed 42

# mmlu ttt iter15 save seed43 run3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag mmlu_ttt_iter15_save_seed43_run3 \
    --ttt_iters 15 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 250 \
    --ttt_save \
    --seed 43

# mmlu ttt iter15 save seed44 run3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag mmlu_ttt_iter15_save_seed44_run3 \
    --ttt_iters 15 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 250 \
    --ttt_save \
    --seed 44

# mmlu ttt iter15 save seed45 run3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag mmlu_ttt_iter15_save_seed45_run3 \
    --ttt_iters 15 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 250 \
    --ttt_save \
    --seed 45

# mmlu ttt iter15 save seed46 run3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag mmlu_ttt_iter15_save_seed46_run3 \
    --ttt_iters 15 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 250 \
    --ttt_save \
    --seed 46

# 46.99523156166195
# 45.735023237925205
# 42.466742349681255
# 45.88585625584651
# 44.29532551023613