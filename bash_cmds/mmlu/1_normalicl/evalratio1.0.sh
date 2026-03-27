# 40mins each

# mmlu normalicl seed42 evalratio1.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_normalicl_seed42_evalratio1.0 \
    --eval_ratio 1.0 \
    --seed 42

# mmlu normalicl seed43 evalratio1.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_normalicl_seed43_evalratio1.0 \
    --eval_ratio 1.0 \
    --seed 43

# mmlu normalicl seed44 evalratio1.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_normalicl_seed44_evalratio1.0 \
    --eval_ratio 1.0 \
    --seed 44

# mmlu normalicl seed45 evalratio1.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_normalicl_seed45_evalratio1.0 \
    --eval_ratio 1.0 \
    --seed 45

# mmlu normalicl seed46 evalratio1.0
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_normalicl_seed46_evalratio1.0 \
    --eval_ratio 1.0 \
    --seed 46

# 40.720762019972234
# 40.854922862379176
# 41.97281893099935
# 41.01580416545961
# 41.47341565681671

# avg: 41.207544727125