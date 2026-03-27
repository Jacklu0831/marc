# bbh gs8 lr2e-3 tokendrop0.2 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr2e-3_tokendrop0.2_seed43 \
    --gs_epochs 8 \
    --gs_lr 2e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.2 \
    --seed 43

# bbh gs12 lr2e-3 tokendrop0.2 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr2e-3_tokendrop0.2_seed43 \
    --gs_epochs 12 \
    --gs_lr 2e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.2 \
    --seed 43

# bbh gs16 lr2e-3 tokendrop0.2 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr2e-3_tokendrop0.2_seed43 \
    --gs_epochs 16 \
    --gs_lr 2e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.2 \
    --seed 43

# bbh gs20 lr2e-3 tokendrop0.2 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr2e-3_tokendrop0.2_seed43 \
    --gs_epochs 20 \
    --gs_lr 2e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.2 \
    --seed 43

# 55.811437275713494
# 56.13200877740655
# 55.78201461307415
# 55.96674161632524