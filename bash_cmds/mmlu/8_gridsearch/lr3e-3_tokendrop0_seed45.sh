# mmlu gs5 lr3e-3 tokendrop0 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr3e-3_tokendrop0_seed45 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0 \
    --seed 45

# mmlu gs10 lr3e-3 tokendrop0 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr3e-3_tokendrop0_seed45 \
    --gs_epochs 10 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0 \
    --seed 45

# mmlu gs15 lr3e-3 tokendrop0 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr3e-3_tokendrop0_seed45 \
    --gs_epochs 15 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0 \
    --seed 45

# mmlu gs20 lr3e-3 tokendrop0 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr3e-3_tokendrop0_seed45 \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0 \
    --seed 45

# mmlu gs25 lr3e-3 tokendrop0 seed45
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr3e-3_tokendrop0_seed45 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0 \
    --seed 45

# 44.81421768741065
# 43.3568198502152
# 41.11807196734091
# 39.145584812764675
# 38.03806629847022