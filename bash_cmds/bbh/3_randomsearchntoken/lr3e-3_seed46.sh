# all under 3hrs

# bbh gs8 lr3e-3 randomkv token ntoken32 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs8_lr3e-3_randomkv_token_ntoken32_seed46 \
    --gs_epochs 8 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --seed 46

# bbh gs12 lr3e-3 randomkv token ntoken32 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs12_lr3e-3_randomkv_token_ntoken32_seed46 \
    --gs_epochs 12 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --seed 46

# bbh gs16 lr3e-3 randomkv token ntoken32 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs16_lr3e-3_randomkv_token_ntoken32_seed46 \
    --gs_epochs 16 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --seed 46

# bbh gs20 lr3e-3 randomkv token ntoken32 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs20_lr3e-3_randomkv_token_ntoken32_seed46 \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --seed 46
