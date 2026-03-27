# bbh gs8 lr1e-1 tokendrop0.05 seed42 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr1e-1_tokendrop0.05_seed42_ntoken32 \
    --gs_epochs 8 \
    --gs_lr 1e-1 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --seed 42

# bbh gs12 lr1e-1 tokendrop0.05 seed42 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr1e-1_tokendrop0.05_seed42_ntoken32 \
    --gs_epochs 12 \
    --gs_lr 1e-1 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --seed 42

# bbh gs16 lr1e-1 tokendrop0.05 seed42 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr1e-1_tokendrop0.05_seed42_ntoken32 \
    --gs_epochs 16 \
    --gs_lr 1e-1 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --seed 42

# bbh gs20 lr1e-1 tokendrop0.05 seed42 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr1e-1_tokendrop0.05_seed42_ntoken32 \
    --gs_epochs 20 \
    --gs_lr 1e-1 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --seed 42

# 54.598265423007696
# 54.98279230076911
# 51.877045491736204
# 54.28607224676811