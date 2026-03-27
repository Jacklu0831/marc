# bbh gs8 lr3e-3 tokendrop0.05 nokey seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr3e-3_tokendrop0.05_nokey_seed45 \
    --gs_epochs 8 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.05 \
    --gs_no_key \
    --seed 45

# bbh gs12 lr3e-3 tokendrop0.05 nokey seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr3e-3_tokendrop0.05_nokey_seed45 \
    --gs_epochs 12 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.05 \
    --gs_no_key \
    --seed 45

# bbh gs16 lr3e-3 tokendrop0.05 nokey seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr3e-3_tokendrop0.05_nokey_seed45 \
    --gs_epochs 16 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.05 \
    --gs_no_key \
    --seed 45

# bbh gs20 lr3e-3 tokendrop0.05 nokey seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr3e-3_tokendrop0.05_nokey_seed45 \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.05 \
    --gs_no_key \
    --seed 45

# 57.03796736085739
# 56.34773474607234
# 55.14934234563391
# 55.03136037023505