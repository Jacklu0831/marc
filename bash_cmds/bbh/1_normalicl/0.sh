accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_normalicl_seed42 \
    --seed 42

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_normalicl_seed43 \
    --seed 43

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_normalicl_seed44 \
    --seed 44

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_normalicl_seed45 \
    --seed 45

accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_normalicl_seed46 \
    --seed 46

# maxseqlen 4096
# 47.10017654268236
# 48.81980821204816
# 48.626277252897395
# 48.4337463799079
# 47.5285782863051
# avg: 48.101717334768

# maxseqlen 2048
# 48.16493822614956
# 50.4496452195336
# 49.671636198338994
# 49.623218852758484
# 48.854312558356675
# avg: 49.352750211027