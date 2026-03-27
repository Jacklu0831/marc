# bbh gs8 lr1e-3 tokendrop0 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr1e-3_tokendrop0_seed43 \
    --gs_epochs 8 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0 \
    --seed 43

# bbh gs12 lr1e-3 tokendrop0 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr1e-3_tokendrop0_seed43 \
    --gs_epochs 12 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0 \
    --seed 43

# bbh gs16 lr1e-3 tokendrop0 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr1e-3_tokendrop0_seed43 \
    --gs_epochs 16 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0 \
    --seed 43

# bbh gs20 lr1e-3 tokendrop0 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr1e-3_tokendrop0_seed43 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0 \
    --seed 43

# 55.93260296086304
# 56.18457197135882
# 55.88024190599935
# 55.82567841393587