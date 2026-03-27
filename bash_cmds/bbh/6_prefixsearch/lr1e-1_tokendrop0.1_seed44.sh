# bbh gs8 lr1e-1 tokendrop0.1 seed44 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr1e-1_tokendrop0.1_seed44_ntoken32 \
    --gs_epochs 8 \
    --gs_lr 1e-1 \
    --gs_token_dropout 0.1 \
    --gs_ntokens 32 \
    --seed 44

# bbh gs12 lr1e-1 tokendrop0.1 seed44 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr1e-1_tokendrop0.1_seed44_ntoken32 \
    --gs_epochs 12 \
    --gs_lr 1e-1 \
    --gs_token_dropout 0.1 \
    --gs_ntokens 32 \
    --seed 44

# bbh gs16 lr1e-1 tokendrop0.1 seed44 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr1e-1_tokendrop0.1_seed44_ntoken32 \
    --gs_epochs 16 \
    --gs_lr 1e-1 \
    --gs_token_dropout 0.1 \
    --gs_ntokens 32 \
    --seed 44

# bbh gs20 lr1e-1 tokendrop0.1 seed44 ntoken32
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr1e-1_tokendrop0.1_seed44_ntoken32 \
    --gs_epochs 20 \
    --gs_lr 1e-1 \
    --gs_token_dropout 0.1 \
    --gs_ntokens 32 \
    --seed 44

# 45.36475226055335
# 51.98696803282716
# 52.599283134306354
# 50.02732781709175