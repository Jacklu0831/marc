# 10mins each

# mmlu zeroshot seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_zeroshot_seed42 \
    --zero_shot \
    --seed 42

# mmlu zeroshot seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_zeroshot_seed43 \
    --zero_shot \
    --seed 43

# mmlu zeroshot seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_zeroshot_seed44 \
    --zero_shot \
    --seed 44

# mmlu zeroshot seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_zeroshot_seed45 \
    --zero_shot \
    --seed 45

# mmlu zeroshot seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_zeroshot_seed46 \
    --zero_shot \
    --seed 46

# OLD
# 35.226941130450015
# 35.6936990937187
# 34.85411696395972
# 36.067829389917854
# 34.00557986156638
# avg: 35.169633287923

# NEW:
# 35.226941130450015
# 35.6936990937187
# 34.85411696395972
# 36.067829389917854
# 34.00557986156638