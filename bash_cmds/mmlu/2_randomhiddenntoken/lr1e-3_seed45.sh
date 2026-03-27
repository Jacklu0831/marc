# this finds that iter25 is best
# all under 2hrs

# mmlu llama8b gs25 lr1e-3 randomhidden token ntoken32 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_hidden/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_randomhidden_token_ntoken32_seed45 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_hidden token \
    --random_hidden_ntokens 32 \
    --seed 45

# mmlu llama8b gs50 lr1e-3 randomhidden token ntoken32 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_hidden/test_time_evaluate.py \
    --tag mmlu_gs50_lr1e-3_randomhidden_token_ntoken32_seed45 \
    --gs_epochs 50 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_hidden token \
    --random_hidden_ntokens 32 \
    --seed 45

# mmlu llama8b gs75 lr1e-3 randomhidden token ntoken32 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_hidden/test_time_evaluate.py \
    --tag mmlu_gs75_lr1e-3_randomhidden_token_ntoken32_seed45 \
    --gs_epochs 75 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_hidden token \
    --random_hidden_ntokens 32 \
    --seed 45

# mmlu llama8b gs100 lr1e-3 randomhidden token ntoken32 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu_hidden/test_time_evaluate.py \
    --tag mmlu_gs100_lr1e-3_randomhidden_token_ntoken32_seed45 \
    --gs_epochs 100 \
    --gs_lr 1e-3 \
    --gs_dropout none \
    --random_hidden token \
    --random_hidden_ntokens 32 \
    --seed 45

# 39.790526096472966 <-
# 36.54992489393481
# 34.743509804549866
# 35.175305802215476