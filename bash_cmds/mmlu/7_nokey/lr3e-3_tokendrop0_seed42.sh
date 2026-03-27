# mmlu gs5 lr3e-3 nokey tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr3e-3_nokey_tokendrop0_seed42 \
    --gs_epochs 5 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 42

# mmlu gs10 lr3e-3 nokey tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr3e-3_nokey_tokendrop0_seed42 \
    --gs_epochs 10 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 42

# mmlu gs15 lr3e-3 nokey tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr3e-3_nokey_tokendrop0_seed42 \
    --gs_epochs 15 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 42

# mmlu gs20 lr3e-3 nokey tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr3e-3_nokey_tokendrop0_seed42 \
    --gs_epochs 20 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 42

# mmlu gs25 lr3e-3 nokey tokendrop0 seed42
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr3e-3_nokey_tokendrop0_seed42 \
    --gs_epochs 25 \
    --gs_lr 3e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 42

# 42.938025328558886
# 42.73927472639466
# 41.07707179303895
# 40.02989533143268
# 39.5789691433387