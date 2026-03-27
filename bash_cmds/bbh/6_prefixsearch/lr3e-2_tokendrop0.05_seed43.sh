# bbh gs8 lr3e-2 tokendrop0.05 seed43 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr3e-2_tokendrop0.05_seed43_ntoken32 \
    --gs_epochs 8 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --seed 43

# bbh gs12 lr3e-2 tokendrop0.05 seed43 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr3e-2_tokendrop0.05_seed43_ntoken32 \
    --gs_epochs 12 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --seed 43

# bbh gs16 lr3e-2 tokendrop0.05 seed43 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr3e-2_tokendrop0.05_seed43_ntoken32 \
    --gs_epochs 16 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --seed 43

# bbh gs20 lr3e-2 tokendrop0.05 seed43 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr3e-2_tokendrop0.05_seed43_ntoken32 \
    --gs_epochs 20 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --seed 43

# 49.52193402461386
# 50.0303665642964
# 51.37157699425151
# 51.598598525755676