# mmlu ttt iter15 save seed0 run3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag mmlu_ttt_iter15_save_seed0_run3 \
    --ttt_iters 15 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 250 \
    --ttt_save \
    --seed 0

# mmlu ttt iter15 save seed1 run3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag mmlu_ttt_iter15_save_seed1_run3 \
    --ttt_iters 15 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 250 \
    --ttt_save \
    --seed 1

# mmlu ttt iter15 save seed2 run3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag mmlu_ttt_iter15_save_seed2_run3 \
    --ttt_iters 15 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 250 \
    --ttt_save \
    --seed 2

# mmlu ttt iter15 save seed3 run3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag mmlu_ttt_iter15_save_seed3_run3 \
    --ttt_iters 15 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 250 \
    --ttt_save \
    --seed 3

# mmlu ttt iter15 save seed4 run3
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_seed/test_time_evaluate.py \
    --tag mmlu_ttt_iter15_save_seed4_run3 \
    --ttt_iters 15 \
    --ttt_gradient_checkpointing \
    --ttt_permute_n 250 \
    --ttt_save \
    --seed 4

# 45.94434616413889
# 42.497417523911984
# 48.21726485771906
# 43.746110226788126
# 44.91375410262486