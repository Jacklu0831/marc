# all under 2hrs

# bbh gs50 lr1e-3 randomhidden token ntoken32 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_gs50_lr1e-3_randomhidden_token_ntoken32_seed42 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_hidden token \
    --random_hidden_ntokens 32 \
    --seed 42

# bbh gs50 lr1e-3 randomhidden token ntoken32 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_gs50_lr1e-3_randomhidden_token_ntoken32_seed43 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_hidden token \
    --random_hidden_ntokens 32 \
    --seed 43

# bbh gs50 lr1e-3 randomhidden token ntoken32 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_gs50_lr1e-3_randomhidden_token_ntoken32_seed44 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_hidden token \
    --random_hidden_ntokens 32 \
    --seed 44

# bbh gs50 lr1e-3 randomhidden token ntoken32 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh_hidden/test_time_evaluate.py \
    --tag bbh_gs50_lr1e-3_randomhidden_token_ntoken32_seed46 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_hidden token \
    --random_hidden_ntokens 32 \
    --seed 46

# 41.63419448535428
# 42.13868604670785
# 46.88221779940046
# 47.68177336724164