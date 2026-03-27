# bbh gs8 lr2e-3 tokendrop0.2 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr2e-3_tokendrop0.2_seed44 \
    --gs_epochs 8 \
    --gs_lr 2e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.2 \
    --seed 44

# bbh gs12 lr2e-3 tokendrop0.2 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr2e-3_tokendrop0.2_seed44 \
    --gs_epochs 12 \
    --gs_lr 2e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.2 \
    --seed 44

# bbh gs16 lr2e-3 tokendrop0.2 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr2e-3_tokendrop0.2_seed44 \
    --gs_epochs 16 \
    --gs_lr 2e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.2 \
    --seed 44

# bbh gs20 lr2e-3 tokendrop0.2 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr2e-3_tokendrop0.2_seed44 \
    --gs_epochs 20 \
    --gs_lr 2e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.2 \
    --seed 44

# 54.777300174455746
# 54.515992800629014
# 56.006007666224384
# 56.588123187871645