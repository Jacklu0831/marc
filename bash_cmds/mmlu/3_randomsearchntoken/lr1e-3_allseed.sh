# now that we know iter25 is best, search all
# all under 2 hrs

# mmlu llama8b gs25 lr1e-3 randomkv token ntoken32 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_randomkv_token_ntoken32_seed42 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --seed 42

# mmlu llama8b gs25 lr1e-3 randomkv token ntoken32 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_randomkv_token_ntoken32_seed43 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --seed 43

# mmlu llama8b gs25 lr1e-3 randomkv token ntoken32 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_randomkv_token_ntoken32_seed44 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --seed 44

# mmlu llama8b gs25 lr1e-3 randomkv token ntoken32 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_randomkv_token_ntoken32_seed46 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_kv token \
    --random_kv_ntokens 32 \
    --seed 46

# 39.368597895439954
# 38.866826323614184
# 41.0568303712721
# 41.04997142934431 # seed45
# 39.27208870970997