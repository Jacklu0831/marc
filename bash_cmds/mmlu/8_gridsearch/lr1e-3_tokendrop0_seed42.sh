# mmlu gs5 lr1e-3 tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr1e-3_tokendrop0_seed42 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0 \
    --seed 42

# mmlu gs10 lr1e-3 tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr1e-3_tokendrop0_seed42 \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0 \
    --seed 42

# mmlu gs15 lr1e-3 tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-3_tokendrop0_seed42 \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0 \
    --seed 42

# mmlu gs20 lr1e-3 tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-3_tokendrop0_seed42 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0 \
    --seed 42

# mmlu gs25 lr1e-3 tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_tokendrop0_seed42 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0 \
    --seed 42

# 41.382226231469225
# 41.39688435670351
# 42.38226823815258
# 42.87212839530061
# 43.18134220014418