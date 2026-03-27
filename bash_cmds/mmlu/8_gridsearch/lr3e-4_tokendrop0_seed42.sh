# mmlu gs5 lr3e-4 tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr3e-4_tokendrop0_seed42 \
    --gs_epochs 5 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0 \
    --seed 42

# mmlu gs10 lr3e-4 tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr3e-4_tokendrop0_seed42 \
    --gs_epochs 10 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0 \
    --seed 42

# mmlu gs15 lr3e-4 tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr3e-4_tokendrop0_seed42 \
    --gs_epochs 15 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0 \
    --seed 42

# mmlu gs20 lr3e-4 tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr3e-4_tokendrop0_seed42 \
    --gs_epochs 20 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0 \
    --seed 42

# mmlu gs25 lr3e-4 tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr3e-4_tokendrop0_seed42 \
    --gs_epochs 25 \
    --gs_lr 3e-4 \
    --gs_token_dropout 0 \
    --seed 42

# 40.916674870015555
# 41.38603826929601
# 41.80754649656255
# 41.71003908921567
# 41.69013926138255