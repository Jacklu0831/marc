# bbh gs8 lr1e-3 tokendrop0.2 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr1e-3_tokendrop0.2_seed45 \
    --gs_epochs 8 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.2 \
    --seed 45

# bbh gs12 lr1e-3 tokendrop0.2 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr1e-3_tokendrop0.2_seed45 \
    --gs_epochs 12 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.2 \
    --seed 45

# bbh gs16 lr1e-3 tokendrop0.2 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr1e-3_tokendrop0.2_seed45 \
    --gs_epochs 16 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.2 \
    --seed 45

# bbh gs20 lr1e-3 tokendrop0.2 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr1e-3_tokendrop0.2_seed45 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_batch_size 2 \
    --gs_token_drop 0.2 \
    --seed 45

# 56.69787480209475
# 57.60306296431617
# 57.97664718061138
# 58.35906101571063