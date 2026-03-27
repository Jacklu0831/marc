# all under 3hrs

# bbh gs25 lr1e-3 randomkv token seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs25_lr1e-3_randomkv_token_seed45 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --seed 45

# bbh gs50 lr1e-3 randomkv token seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs50_lr1e-3_randomkv_token_seed45 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --seed 45

# bbh gs75 lr1e-3 randomkv token seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs75_lr1e-3_randomkv_token_seed45 \
    --gs_epochs 75 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --seed 45

# bbh gs100 lr1e-3 randomkv token seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_bbh/test_time_evaluate.py \
    --tag bbh_gs100_lr1e-3_randomkv_token_seed45 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --seed 45

# 54.62991718426501
# 54.2802338326635
# 53.645871392035076
# 53.81667884545123