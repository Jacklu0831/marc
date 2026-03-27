# now that we know iter25 is best, search all
# all under 2hrs

# mmlu llama8b gs25 lr1e-3 randomhidden token ntoken32 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_hidden/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_randomhidden_token_ntoken32_seed42 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_hidden token \
    --random_hidden_ntokens 32 \
    --seed 42

# mmlu llama8b gs25 lr1e-3 randomhidden token ntoken32 seed43
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_hidden/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_randomhidden_token_ntoken32_seed43 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_hidden token \
    --random_hidden_ntokens 32 \
    --seed 43

# mmlu llama8b gs25 lr1e-3 randomhidden token ntoken32 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_hidden/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_randomhidden_token_ntoken32_seed44 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_hidden token \
    --random_hidden_ntokens 32 \
    --seed 44

# mmlu llama8b gs25 lr1e-3 randomhidden token ntoken32 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_hidden/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_randomhidden_token_ntoken32_seed46 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_hidden token \
    --random_hidden_ntokens 32 \
    --seed 46

# 37.70470407395038
# 38.67771335302495
# 38.294967458349625
# 39.790526096472966 # seed45
# 38.138207892716075
# avg: 38.521223774903