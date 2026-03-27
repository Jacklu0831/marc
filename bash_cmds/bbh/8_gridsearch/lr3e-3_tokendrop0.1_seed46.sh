# bbh gs8 lr3e-3 tokendrop0.1 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr3e-3_tokendrop0.1_seed46 \
    --gs_epochs 8 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.1 \
    --seed 46

# bbh gs12 lr3e-3 tokendrop0.1 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr3e-3_tokendrop0.1_seed46 \
    --gs_epochs 12 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.1 \
    --seed 46

# bbh gs16 lr3e-3 tokendrop0.1 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr3e-3_tokendrop0.1_seed46 \
    --gs_epochs 16 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.1 \
    --seed 46

# bbh gs20 lr3e-3 tokendrop0.1 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr3e-3_tokendrop0.1_seed46 \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.1 \
    --seed 46

# 54.40744938326207
# 53.83803411715564
# 52.81335231461005
# 53.62740490933215