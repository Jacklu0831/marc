# 40mins each

# mmlu zeroshot seed42 evalratio1.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_zeroshot_seed42_evalratio1.0 \
    --zero_shot \
    --eval_ratio 1.0 \
    --seed 42

# mmlu zeroshot seed43 evalratio1.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_zeroshot_seed43_evalratio1.0 \
    --zero_shot \
    --eval_ratio 1.0 \
    --seed 43

# mmlu zeroshot seed44 evalratio1.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_zeroshot_seed44_evalratio1.0 \
    --zero_shot \
    --eval_ratio 1.0 \
    --seed 44

# mmlu zeroshot seed45 evalratio1.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_zeroshot_seed45_evalratio1.0 \
    --zero_shot \
    --eval_ratio 1.0 \
    --seed 45

# mmlu zeroshot seed46 evalratio1.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_zeroshot_seed46_evalratio1.0 \
    --zero_shot \
    --eval_ratio 1.0 \
    --seed 46

# 35.625139300457015
# 35.71047971160164
# 35.93100280977245
# 35.922738333312594
# 35.95085905501726
# avg: 35.828043842032