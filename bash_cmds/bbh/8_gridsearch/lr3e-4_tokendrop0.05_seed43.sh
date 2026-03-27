# bbh gs8 lr3e-4 tokendrop0.05 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr3e-4_tokendrop0.05_seed43 \
    --gs_epochs 8 \
    --gs_lr 3e-4 \
    --gs_batch_size 2 \
    --gs_token_drop 0.05 \
    --seed 43

# bbh gs12 lr3e-4 tokendrop0.05 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr3e-4_tokendrop0.05_seed43 \
    --gs_epochs 12 \
    --gs_lr 3e-4 \
    --gs_batch_size 2 \
    --gs_token_drop 0.05 \
    --seed 43

# bbh gs16 lr3e-4 tokendrop0.05 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr3e-4_tokendrop0.05_seed43 \
    --gs_epochs 16 \
    --gs_lr 3e-4 \
    --gs_batch_size 2 \
    --gs_token_drop 0.05 \
    --seed 43

# bbh gs20 lr3e-4 tokendrop0.05 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr3e-4_tokendrop0.05_seed43 \
    --gs_epochs 20 \
    --gs_lr 3e-4 \
    --gs_batch_size 2 \
    --gs_token_drop 0.05 \
    --seed 43

# 51.65184899260908
# 53.13405231570539
# 54.51104839526172
# 55.38245945968749