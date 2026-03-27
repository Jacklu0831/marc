# bbh gs8 lr3e-3 tokendrop0.2 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr3e-3_tokendrop0.2_seed42 \
    --gs_epochs 8 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.2 \
    --seed 42

# bbh gs12 lr3e-3 tokendrop0.2 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr3e-3_tokendrop0.2_seed42 \
    --gs_epochs 12 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.2 \
    --seed 42

# bbh gs16 lr3e-3 tokendrop0.2 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr3e-3_tokendrop0.2_seed42 \
    --gs_epochs 16 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.2 \
    --seed 42

# bbh gs20 lr3e-3 tokendrop0.2 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr3e-3_tokendrop0.2_seed42 \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.2 \
    --seed 42

# 54.92188778432335
# 56.22080060546555
# 54.86246624938635
# 54.77486806578301