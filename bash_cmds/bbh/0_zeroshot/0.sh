accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_zeroshot_seed42 \
    --zero_shot \
    --seed 42

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_zeroshot_seed43 \
    --zero_shot \
    --seed 43

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_zeroshot_seed44 \
    --zero_shot \
    --seed 44

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_zeroshot_seed45 \
    --zero_shot \
    --seed 45

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_zeroshot_seed46 \
    --zero_shot \
    --seed 46

# 32.47724390443463
# 33.560395292655215
# 33.35919455501499
# 32.92032030203386
# 33.59632997936017
# avg: 33.1826968067??? should be 40