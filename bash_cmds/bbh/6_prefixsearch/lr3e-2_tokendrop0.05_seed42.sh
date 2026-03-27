# bbh gs8 lr3e-2 tokendrop0.05 seed42 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr3e-2_tokendrop0.05_seed42_ntoken32 \
    --gs_epochs 8 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --seed 42

# bbh gs12 lr3e-2 tokendrop0.05 seed42 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr3e-2_tokendrop0.05_seed42_ntoken32 \
    --gs_epochs 12 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --seed 42

# bbh gs16 lr3e-2 tokendrop0.05 seed42 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr3e-2_tokendrop0.05_seed42_ntoken32 \
    --gs_epochs 16 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --seed 42

# bbh gs20 lr3e-2 tokendrop0.05 seed42 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr3e-2_tokendrop0.05_seed42_ntoken32 \
    --gs_epochs 20 \
    --gs_lr 3e-2 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --seed 42

# 49.19747688594337
# 50.45739752086401
# 51.51811282932416
# 52.91275466372116