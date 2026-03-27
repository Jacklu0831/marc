# 10mins each

# mmlu normalicl seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_normalicl_seed42 \
    --seed 42

# mmlu normalicl seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_normalicl_seed43 \
    --seed 43

# mmlu normalicl seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_normalicl_seed44 \
    --seed 44

# mmlu normalicl seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_normalicl_seed45 \
    --seed 45

# mmlu normalicl seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_normalicl_seed46 \
    --seed 46

# 40.67055665601556
# 41.248353618133756
# 42.697754005810914
# 41.47181653684334
# 41.84622112522983
# avg: 41.586940388407

# 00:28:46
# 00:37:30
# 00:46:07
# 00:54:47
# 01:03:37
# 01:11:59
# ~8-9mins