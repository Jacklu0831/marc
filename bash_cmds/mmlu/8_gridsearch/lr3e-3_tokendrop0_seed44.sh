# mmlu gs5 lr3e-3 tokendrop0 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr3e-3_tokendrop0_seed44 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0 \
    --seed 44

# mmlu gs10 lr3e-3 tokendrop0 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr3e-3_tokendrop0_seed44 \
    --gs_epochs 10 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0 \
    --seed 44

# mmlu gs15 lr3e-3 tokendrop0 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr3e-3_tokendrop0_seed44 \
    --gs_epochs 15 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0 \
    --seed 44

# mmlu gs20 lr3e-3 tokendrop0 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr3e-3_tokendrop0_seed44 \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0 \
    --seed 44

# mmlu gs25 lr3e-3 tokendrop0 seed44
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr3e-3_tokendrop0_seed44 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0 \
    --seed 44

# 43.8726721984578
# 43.33836966922107
# 42.41296985143803
# 41.168945691137395
# 39.170019846242894