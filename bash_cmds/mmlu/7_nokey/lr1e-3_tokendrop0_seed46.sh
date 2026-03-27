# mmlu gs5 lr1e-3 nokey tokendrop0 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs5_lr1e-3_nokey_tokendrop0_seed46 \
    --gs_epochs 5 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 46

# mmlu gs10 lr1e-3 nokey tokendrop0 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs10_lr1e-3_nokey_tokendrop0_seed46 \
    --gs_epochs 10 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 46

# mmlu gs15 lr1e-3 nokey tokendrop0 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs15_lr1e-3_nokey_tokendrop0_seed46 \
    --gs_epochs 15 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 46

# mmlu gs20 lr1e-3 nokey tokendrop0 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs20_lr1e-3_nokey_tokendrop0_seed46 \
    --gs_epochs 20 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 46

# mmlu gs25 lr1e-3 nokey tokendrop0 seed46
accelerate launch --main_process_port $MASTER_PORT --mixed_precision bf16 inference_mmlu/test_time_evaluate.py \
    --tag mmlu_gs25_lr1e-3_nokey_tokendrop0_seed46 \
    --gs_epochs 25 \
    --gs_lr 1e-3 \
    --gs_token_dropout 0 \
    --gs_no_key \
    --seed 46

# 42.557850515715856
# 42.60530140195598
# 42.678133291106896
# 42.8403460901224
# 42.716538749564386