# bbh gs8 lr3e-4 tokendrop0 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr3e-4_tokendrop0_seed45 \
    --gs_epochs 8 \
    --gs_lr 3e-4 \
    --gs_batch_size 2 \
    --gs_token_drop 0 \
    --seed 45

# bbh gs12 lr3e-4 tokendrop0 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr3e-4_tokendrop0_seed45 \
    --gs_epochs 12 \
    --gs_lr 3e-4 \
    --gs_batch_size 2 \
    --gs_token_drop 0 \
    --seed 45

# bbh gs16 lr3e-4 tokendrop0 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr3e-4_tokendrop0_seed45 \
    --gs_epochs 16 \
    --gs_lr 3e-4 \
    --gs_batch_size 2 \
    --gs_token_drop 0 \
    --seed 45

# bbh gs20 lr3e-4 tokendrop0 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr3e-4_tokendrop0_seed45 \
    --gs_epochs 20 \
    --gs_lr 3e-4 \
    --gs_batch_size 2 \
    --gs_token_drop 0 \
    --seed 45

# 53.30273413713311
# 55.12331019364269
# 56.58994032395567
# 57.29661429789308