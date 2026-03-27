# bbh gs8 lr3e-2 tokendrop0 seed43 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr3e-2_tokendrop0_seed43_ntoken32 \
    --gs_epochs 8 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0 \
    --gs_ntokens 32 \
    --seed 43

# bbh gs12 lr3e-2 tokendrop0 seed43 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr3e-2_tokendrop0_seed43_ntoken32 \
    --gs_epochs 12 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0 \
    --gs_ntokens 32 \
    --seed 43

# bbh gs16 lr3e-2 tokendrop0 seed43 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr3e-2_tokendrop0_seed43_ntoken32 \
    --gs_epochs 16 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0 \
    --gs_ntokens 32 \
    --seed 43

# bbh gs20 lr3e-2 tokendrop0 seed43 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr3e-2_tokendrop0_seed43_ntoken32 \
    --gs_epochs 20 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0 \
    --gs_ntokens 32 \
    --seed 43

# 49.41951805822731
# 50.46646494943358
# 51.81216989807971
# 53.09909455750172