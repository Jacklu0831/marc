# all under 3hrs

# bbh gs25 lr1e-3 randomhidden token ntoken32 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_gs25_lr1e-3_randomhidden_token_ntoken32_seed45 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_hidden token \
    --random_hidden_ntokens 32 \
    --seed 45

# bbh gs50 lr1e-3 randomhidden token ntoken32 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_gs50_lr1e-3_randomhidden_token_ntoken32_seed45 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_hidden token \
    --random_hidden_ntokens 32 \
    --seed 45

# bbh gs75 lr1e-3 randomhidden token ntoken32 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_gs75_lr1e-3_randomhidden_token_ntoken32_seed45 \
    --gs_epochs 75 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_hidden token \
    --random_hidden_ntokens 32 \
    --seed 45

# bbh gs100 lr1e-3 randomhidden token ntoken32 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_gs100_lr1e-3_randomhidden_token_ntoken32_seed45 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_hidden token \
    --random_hidden_ntokens 32 \
    --seed 45

# 45.040189989039085
# 52.03066009012299
# 46.31667884545123
# 41.329771038850325