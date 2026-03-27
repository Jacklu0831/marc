# bbh gs8 lr3e-1 tokendrop0 seed45 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr3e-1_tokendrop0_seed45_ntoken32 \
    --gs_epochs 8 \
    --gs_lr 3e-1 \
    --gs_token_dropout 0 \
    --gs_ntokens 32 \
    --seed 45

# bbh gs12 lr3e-1 tokendrop0 seed45 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr3e-1_tokendrop0_seed45_ntoken32 \
    --gs_epochs 12 \
    --gs_lr 3e-1 \
    --gs_token_dropout 0 \
    --gs_ntokens 32 \
    --seed 45

# bbh gs16 lr3e-1 tokendrop0 seed45 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr3e-1_tokendrop0_seed45_ntoken32 \
    --gs_epochs 16 \
    --gs_lr 3e-1 \
    --gs_token_dropout 0 \
    --gs_ntokens 32 \
    --seed 45

# bbh gs20 lr3e-1 tokendrop0 seed45 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr3e-1_tokendrop0_seed45_ntoken32 \
    --gs_epochs 20 \
    --gs_lr 3e-1 \
    --gs_token_dropout 0 \
    --gs_ntokens 32 \
    --seed 45

# 34.53720618682256
# 30.61594202898551
# 25.71398124467178
# 28.09676044330776