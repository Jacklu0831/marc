# bbh evalondemon seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_evalondemon_seed42 \
    --eval_on_demonstrations \
    --seed 42

# bbh evalondemon seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_evalondemon_seed43 \
    --eval_on_demonstrations \
    --seed 43

# bbh evalondemon seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_evalondemon_seed44 \
    --eval_on_demonstrations \
    --seed 44

# bbh evalondemon seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_evalondemon_seed45 \
    --eval_on_demonstrations \
    --seed 45

# bbh evalondemon seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_evalondemon_seed46 \
    --eval_on_demonstrations \
    --seed 46

# originally seed0 seqlen4096 got 83.333% and took <10mins, makes sense because theres only so many pairs

# 85.05952380952381
# 81.80555555555556
# 83.6111111111111
# 84.34782608695652
# 85.83333333333333