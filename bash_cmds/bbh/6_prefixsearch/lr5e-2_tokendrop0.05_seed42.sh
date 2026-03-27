# bbh gs8 lr5e-2 tokendrop0.05 seed42 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr5e-2_tokendrop0.05_seed42_ntoken32 \
    --gs_epochs 8 \
    --gs_lr 5e-2 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --seed 42

# bbh gs12 lr5e-2 tokendrop0.05 seed42 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr5e-2_tokendrop0.05_seed42_ntoken32 \
    --gs_epochs 12 \
    --gs_lr 5e-2 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --seed 42

# bbh gs16 lr5e-2 tokendrop0.05 seed42 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr5e-2_tokendrop0.05_seed42_ntoken32 \
    --gs_epochs 16 \
    --gs_lr 5e-2 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --seed 42

# bbh gs20 lr5e-2 tokendrop0.05 seed42 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr5e-2_tokendrop0.05_seed42_ntoken32 \
    --gs_epochs 20 \
    --gs_lr 5e-2 \
    --gs_token_dropout 0.05 \
    --gs_ntokens 32 \
    --seed 42
