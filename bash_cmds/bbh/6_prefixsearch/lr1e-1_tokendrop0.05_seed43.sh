# bbh gs8 lr1e-1 tokendrop0.05 seed43 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr1e-1_tokendrop0.05_seed43_ntoken32 \
    --gs_epochs 8 \
    --gs_lr 1e-1 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --seed 43

# bbh gs12 lr1e-1 tokendrop0.05 seed43 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr1e-1_tokendrop0.05_seed43_ntoken32 \
    --gs_epochs 12 \
    --gs_lr 1e-1 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --seed 43

# bbh gs16 lr1e-1 tokendrop0.05 seed43 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr1e-1_tokendrop0.05_seed43_ntoken32 \
    --gs_epochs 16 \
    --gs_lr 1e-1 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --seed 43

# bbh gs20 lr1e-1 tokendrop0.05 seed43 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr1e-1_tokendrop0.05_seed43_ntoken32 \
    --gs_epochs 20 \
    --gs_lr 1e-1 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --seed 43

# 51.84166814316089
# 54.019468982372054
# 51.71801351062513
# 49.59032031487294