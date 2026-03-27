# bbh gs8 lr3e-4 tokendrop0 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr3e-4_tokendrop0_seed44 \
    --gs_epochs 8 \
    --gs_lr 3e-4 \
    --gs_batch_size 2 \
    --gs_token_drop 0 \
    --seed 44

# bbh gs12 lr3e-4 tokendrop0 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr3e-4_tokendrop0_seed44 \
    --gs_epochs 12 \
    --gs_lr 3e-4 \
    --gs_batch_size 2 \
    --gs_token_drop 0 \
    --seed 44

# bbh gs16 lr3e-4 tokendrop0 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr3e-4_tokendrop0_seed44 \
    --gs_epochs 16 \
    --gs_lr 3e-4 \
    --gs_batch_size 2 \
    --gs_token_drop 0 \
    --seed 44

# bbh gs20 lr3e-4 tokendrop0 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr3e-4_tokendrop0_seed44 \
    --gs_epochs 20 \
    --gs_lr 3e-4 \
    --gs_batch_size 2 \
    --gs_token_drop 0 \
    --seed 44

# 52.27055844267531
# 53.862090888987176
# 54.46053860140547
# 54.976127512408475