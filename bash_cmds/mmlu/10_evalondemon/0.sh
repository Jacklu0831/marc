# mmlu evalondemon seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_evalondemon_seed42 \
    --eval_on_demonstrations \
    --seed 42

# mmlu evalondemon seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_evalondemon_seed43 \
    --eval_on_demonstrations \
    --seed 43

# mmlu evalondemon seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_evalondemon_seed44 \
    --eval_on_demonstrations \
    --seed 44

# mmlu evalondemon seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_evalondemon_seed45 \
    --eval_on_demonstrations \
    --seed 45

# mmlu evalondemon seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_evalondemon_seed46 \
    --eval_on_demonstrations \
    --seed 46

# originally 92.98245614035088, ran locally tho, but should be fast anyway

# 92.36111111111111
# 93.28703703703704
# 93.98148148148148
# 92.47685185185185
# 93.97727272727273