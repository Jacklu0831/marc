# bbh gs8 lr3e-4 tokendrop0.1 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr3e-4_tokendrop0.1_seed43 \
    --gs_epochs 8 \
    --gs_lr 3e-4 \
    --gs_batch_size 2 \
    --gs_token_drop 0.1 \
    --seed 43

# bbh gs12 lr3e-4 tokendrop0.1 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr3e-4_tokendrop0.1_seed43 \
    --gs_epochs 12 \
    --gs_lr 3e-4 \
    --gs_batch_size 2 \
    --gs_token_drop 0.1 \
    --seed 43

# bbh gs16 lr3e-4 tokendrop0.1 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr3e-4_tokendrop0.1_seed43 \
    --gs_epochs 16 \
    --gs_lr 3e-4 \
    --gs_batch_size 2 \
    --gs_token_drop 0.1 \
    --seed 43

# bbh gs20 lr3e-4 tokendrop0.1 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr3e-4_tokendrop0.1_seed43 \
    --gs_epochs 20 \
    --gs_lr 3e-4 \
    --gs_batch_size 2 \
    --gs_token_drop 0.1 \
    --seed 43

# 51.4616062288369
# 52.61689439832157
# 54.34179997300125
# 54.69997813382383