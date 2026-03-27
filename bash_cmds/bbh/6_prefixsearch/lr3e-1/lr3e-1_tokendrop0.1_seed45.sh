# bbh gs8 lr3e-1 tokendrop0.1 seed45 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr3e-1_tokendrop0.1_seed45_ntoken32 \
    --gs_epochs 8 \
    --gs_lr 3e-1 \
    --gs_token_dropout 0.1 \
    --gs_ntokens 32 \
    --seed 45

# bbh gs12 lr3e-1 tokendrop0.1 seed45 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr3e-1_tokendrop0.1_seed45_ntoken32 \
    --gs_epochs 12 \
    --gs_lr 3e-1 \
    --gs_token_dropout 0.1 \
    --gs_ntokens 32 \
    --seed 45

# bbh gs16 lr3e-1 tokendrop0.1 seed45 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr3e-1_tokendrop0.1_seed45_ntoken32 \
    --gs_epochs 16 \
    --gs_lr 3e-1 \
    --gs_token_dropout 0.1 \
    --gs_ntokens 32 \
    --seed 45

# bbh gs20 lr3e-1 tokendrop0.1 seed45 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr3e-1_tokendrop0.1_seed45_ntoken32 \
    --gs_epochs 20 \
    --gs_lr 3e-1 \
    --gs_token_dropout 0.1 \
    --gs_ntokens 32 \
    --seed 45

# 33.171355498721226
# 24.22101449275362
# 20.724637681159418
# 15.62111801242236