# bbh gs8 lr3e-2 tokendrop0.1 seed45 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr3e-2_tokendrop0.1_seed45_ntoken32 \
    --gs_epochs 8 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.1 \
    --gs_ntokens 32 \
    --seed 45

# bbh gs12 lr3e-2 tokendrop0.1 seed45 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr3e-2_tokendrop0.1_seed45_ntoken32 \
    --gs_epochs 12 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.1 \
    --gs_ntokens 32 \
    --seed 45

# bbh gs16 lr3e-2 tokendrop0.1 seed45 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr3e-2_tokendrop0.1_seed45_ntoken32 \
    --gs_epochs 16 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.1 \
    --gs_ntokens 32 \
    --seed 45

# bbh gs20 lr3e-2 tokendrop0.1 seed45 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr3e-2_tokendrop0.1_seed45_ntoken32 \
    --gs_epochs 20 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.1 \
    --gs_ntokens 32 \
    --seed 45

# 49.17991109487274
# 50.75173547679942
# 51.73943490439653
# 53.51403604920229