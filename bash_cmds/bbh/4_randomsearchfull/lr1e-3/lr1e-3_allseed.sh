# all under 2hrs

# bbh gs25 lr1e-3 randomkv token seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs25_lr1e-3_randomkv_token_seed42 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --seed 42

# bbh gs25 lr1e-3 randomkv token seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs25_lr1e-3_randomkv_token_seed43 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --seed 43

# bbh gs25 lr1e-3 randomkv token seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs25_lr1e-3_randomkv_token_seed44 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --seed 44

# bbh gs25 lr1e-3 randomkv token seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs25_lr1e-3_randomkv_token_seed46 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --seed 46

# 52.63270127638685
# 51.603272333029594
# 51.71109452552951
# 53.631075237112384
