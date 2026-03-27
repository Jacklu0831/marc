# all under 3hrs

# bbh gs25 lr1e-3 randomkv token ntoken32 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs25_lr1e-3_randomkv_token_ntoken32_seed45 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --seed 45

# bbh gs50 lr1e-3 randomkv token ntoken32 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs50_lr1e-3_randomkv_token_ntoken32_seed45 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --seed 45

# bbh gs75 lr1e-3 randomkv token ntoken32 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs75_lr1e-3_randomkv_token_ntoken32_seed45 \
    --gs_epochs 75 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --seed 45

# bbh gs100 lr1e-3 randomkv token ntoken32 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs100_lr1e-3_randomkv_token_ntoken32_seed45 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --seed 45

# 53.12370600414079
# 53.61511996102789
# 53.46258068444769
# 53.14060406771405